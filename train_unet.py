"""
train_unet.py — PC-LiquidGAN with UNet-ODE Architecture
===========================================================================
This uses the ODEUNetGenerator, which adds skip connections from the encoder
to the decoder, while running the Neural ODE on the spatial bottleneck.
This physically preserves high-frequency spatial details (edges, textures) 
that were being destroyed by the 128-dim flat latent space in the old model.

Includes all stable training techniques:
1. Instance noise (σ=0.1 → 0)
2. Label smoothing (0.9 / 0.05)
3. 10-epoch L1 warmup
4. 2:1 G:D updates
5. Spectral loss = 0.05

Usage:
    python train_unet.py --dataset agri --epochs 100
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.generator import ODEUNetGenerator
from models.discriminator import LiquidDiscriminator
from losses.physics_loss import PhysicsLoss
from losses.spectral_loss import SpectralLoss
from utils.dataset import ThermalDataset
from utils.metrics import compute_ssim, compute_psnr
from config import Config


def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname and classname != 'ConvODEFunc':
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif 'BatchNorm' in classname or 'InstanceNorm' in classname:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


def add_instance_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    return x + sigma * torch.randn_like(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',    type=str, required=True,
                   choices=['kaist', 'cbsr', 'medical', 'agri', 'chilli'])
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--warmup',     type=int, default=10)
    p.add_argument('--lambda_spec', type=float, default=0.05)
    p.add_argument('--init_noise',  type=float, default=0.1)
    args = p.parse_args()

    cfg = Config()
    cfg.BATCH_SIZE = args.batch_size
    cfg.EPOCHS     = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = f'./results_unet/{args.dataset}'
    ckpt_dir = f'./checkpoints_unet/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  PC-LiquidGAN ODE-UNET Training")
    print(f"  Dataset:        {args.dataset.upper()}")
    print(f"  Architecture:   ODEUNetGenerator (Skip Connections)")
    print(f"  ODE Solver:     Euler (spatial ODE map)")
    print(f"{'='*65}\n")

    base_path  = os.path.join(cfg.DATA_DIR, args.dataset)
    train_path = os.path.join(base_path, 'train')
    data_path  = train_path if os.path.isdir(train_path) else base_path

    dataset = ThermalDataset(
        rgb_dir=os.path.join(data_path, 'rgb'),
        thermal_dir=os.path.join(data_path, 'thermal'),
        img_size=cfg.IMG_SIZE,
    )
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                        num_workers=cfg.NUM_WORKERS,
                        pin_memory=(device.type == 'cuda'))
    print(f"Dataset: {len(dataset)} | Batches/ep: {len(loader)}\n")

    # Models — Using UNet!
    G = ODEUNetGenerator(
        input_channels=3, output_channels=1, ode_method='euler'
    ).to(device)
    D = LiquidDiscriminator(hidden_size=cfg.HIDDEN_SIZE).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    opt_G = optim.Adam(G.parameters(), lr=cfg.LR_G,       betas=(cfg.BETA1, cfg.BETA2))
    opt_D = optim.Adam(D.parameters(), lr=cfg.LR_D * 0.5, betas=(cfg.BETA1, cfg.BETA2))
    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=cfg.EPOCHS)
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=cfg.EPOCHS)

    l1_loss      = nn.L1Loss()
    adv_loss     = nn.BCEWithLogitsLoss()
    physics_fn   = PhysicsLoss(alpha=0.001).to(device)
    spectral_fn  = SpectralLoss(img_size=cfg.IMG_SIZE, sigma=32.0).to(device)

    best_ssim = 0.0

    for epoch in range(cfg.EPOCHS):
        G.train(); D.train()
        t0 = time.time()
        ed, eg, el1, ep, es = 0., 0., 0., 0., 0.

        noise_sigma = args.init_noise * max(0, 1.0 - epoch / 50.0)
        in_warmup = (epoch < args.warmup)

        for rgb, thermal in loader:
            rgb     = rgb.to(device)
            thermal = thermal.to(device)
            B       = rgb.size(0)

            # Discriminator update
            if not in_warmup:
                opt_D.zero_grad()
                fake_t     = G(rgb).detach()
                real_noisy = add_instance_noise(thermal, noise_sigma)
                fake_noisy = add_instance_noise(fake_t,  noise_sigma)
                real_labels = torch.full((B, 1), 0.9,  device=device)
                fake_labels = torch.full((B, 1), 0.05, device=device)

                d_loss = 0.5 * (adv_loss(D(real_noisy), real_labels) +
                                adv_loss(D(fake_noisy), fake_labels))
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.GRAD_CLIP)
                opt_D.step()
                ed += d_loss.item()

            # Generator update (2 steps per D step)
            for _ in range(2 if not in_warmup else 1):
                opt_G.zero_grad()
                fake_t = G(rgb)
                l1 = l1_loss(fake_t, thermal)

                if in_warmup:
                    g_loss = l1
                else:
                    real_labels_g = torch.full((B, 1), 1.0, device=device)
                    fake_noisy_g  = add_instance_noise(fake_t, noise_sigma)
                    l_adv  = adv_loss(D(fake_noisy_g), real_labels_g)
                    l_phys = physics_fn(fake_t, thermal,
                                       lambda_flux=cfg.LAMBDA_FLUX,
                                       lambda_energy=cfg.LAMBDA_ENERGY)
                    l_spec = spectral_fn(fake_t, thermal)
                    g_loss = (cfg.LAMBDA_ADV * l_adv +
                              10.0 * l1 + 
                              0.1  * l_phys +
                              args.lambda_spec * l_spec)
                    ep += l_phys.item()
                    es += l_spec.item()

                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.GRAD_CLIP)
                opt_G.step()
                eg  += g_loss.item()
                el1 += l1.item()

        sched_G.step()
        sched_D.step()

        n = len(loader)
        elapsed = time.time() - t0
        mode_str = "WARMUP" if in_warmup else f"σ={noise_sigma:.3f}"
        print(f"[UNet|{mode_str:10s}] Ep [{epoch+1:3d}/{cfg.EPOCHS}]  "
              f"D:{ed/n:.4f}  G:{eg/n:.4f}  L1:{el1/n:.4f}  "
              f"Phys:{ep/n:.4f}  Spec:{es/n:.4f}  Time:{elapsed:.1f}s")

        if (epoch + 1) % 10 == 0:
            G.eval()
            with torch.no_grad():
                rgb_s, therm_s = next(iter(loader))
                rgb_s   = rgb_s[:4].to(device)
                therm_s = therm_s[:4].to(device)
                fake_s  = G(rgb_s)
                grid = torch.cat([fake_s, therm_s], dim=0)
                save_image(grid * 0.5 + 0.5,
                           os.path.join(save_dir, f'epoch_{epoch+1:04d}.png'),
                           nrow=4)
                ssim_v = compute_ssim(fake_s, therm_s)
                psnr_v = compute_psnr(fake_s, therm_s)
                print(f"  → SSIM: {ssim_v:.4f}  PSNR: {psnr_v:.2f} dB", end='')
                if ssim_v > best_ssim:
                    best_ssim = ssim_v
                    torch.save({
                        'G_state': G.state_dict(),
                        'D_state': D.state_dict(),
                        'epoch': epoch + 1,
                        'best_ssim': best_ssim
                    }, os.path.join(ckpt_dir, 'best.pth'))
                    print(f"  ← Best! Saved.", end='')
                print()
            torch.save({
                'G_state': G.state_dict(),
                'epoch': epoch + 1
            }, os.path.join(ckpt_dir, f'ckpt_epoch_{epoch+1:04d}.pth'))
            G.train()

    print(f"\nBest SSIM: {best_ssim:.4f}")
    print(f"Done! UNet training complete for {args.dataset.upper()}")

if __name__ == '__main__':
    main()
