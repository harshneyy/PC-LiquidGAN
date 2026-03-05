"""
train_ablation.py
Ablation Study: Train PC-LiquidGAN (NeuralODE + LNN) WITHOUT physics loss.
This proves that the physics constraints are essential for high-fidelity synthesis.

Usage:
    python train_ablation.py --dataset agri --epochs 100 --batch_size 32
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.generator import NeuralODEGenerator
from models.discriminator import LiquidDiscriminator
from utils.dataset import ThermalDataset
from utils.metrics import compute_ssim, compute_psnr
from config import Config


def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif 'BatchNorm' in classname or 'InstanceNorm' in classname:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


def main():
    p = argparse.ArgumentParser(description='Ablation: PC-LiquidGAN WITHOUT Physics Loss')
    p.add_argument('--dataset',    type=str, default='agri')
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    args = p.parse_args()

    cfg = Config()
    cfg.BATCH_SIZE = args.batch_size
    cfg.EPOCHS = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  ABLATION STUDY: PC-LiquidGAN WITHOUT Physics Loss")
    print(f"  Dataset: {args.dataset.upper()}")
    print(f"  Device:  {device}")
    print(f"  Physics Loss: DISABLED (lambda_flux=0, lambda_energy=0)")
    print(f"{'='*60}\n")

    # Dataset
    base_path = os.path.join(cfg.DATA_DIR, args.dataset)
    train_path = os.path.join(base_path, 'train')
    if os.path.isdir(train_path):
        data_path = train_path
    else:
        data_path = base_path

    dataset = ThermalDataset(
        rgb_dir=os.path.join(data_path, 'rgb'),
        thermal_dir=os.path.join(data_path, 'thermal'),
        img_size=cfg.IMG_SIZE,
    )
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                        num_workers=cfg.NUM_WORKERS, pin_memory=(device.type == 'cuda'))
    print(f"Dataset: {len(dataset)} pairs | Batches/epoch: {len(loader)}\n")

    # Models (same architecture as full PC-LiquidGAN)
    G = NeuralODEGenerator(latent_dim=cfg.LATENT_DIM, ode_method=cfg.ODE_METHOD).to(device)
    D = LiquidDiscriminator(hidden_size=cfg.HIDDEN_SIZE).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    print(f"Generator params    : {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}\n")

    opt_G = optim.Adam(G.parameters(), lr=cfg.LR_G, betas=(cfg.BETA1, cfg.BETA2))
    opt_D = optim.Adam(D.parameters(), lr=cfg.LR_D, betas=(cfg.BETA1, cfg.BETA2))
    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=cfg.EPOCHS)
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=cfg.EPOCHS)

    adv_loss = nn.BCEWithLogitsLoss()
    rec_loss = nn.L1Loss()
    # NO PHYSICS LOSS — this is the ablation

    save_dir = f'./results_ablation/{args.dataset}'
    ckpt_dir = f'./checkpoints_ablation/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(cfg.EPOCHS):
        G.train()
        D.train()
        t0 = time.time()
        epoch_d, epoch_g = 0., 0.

        for rgb, thermal in loader:
            rgb = rgb.to(device)
            thermal = thermal.to(device)
            B = rgb.size(0)
            real_l = torch.ones(B, 1, device=device)
            fake_l = torch.zeros(B, 1, device=device)

            # ── Discriminator ──
            opt_D.zero_grad()
            fake_thermal = G(rgb).detach()
            d_real = D(thermal)
            d_fake = D(fake_thermal)
            d_loss = 0.5 * (adv_loss(d_real, real_l) + adv_loss(d_fake, fake_l))
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.GRAD_CLIP)
            opt_D.step()

            # ── Generator (NO physics loss!) ──
            opt_G.zero_grad()
            fake_thermal = G(rgb)
            d_fake_for_g = D(fake_thermal)
            l_adv = adv_loss(d_fake_for_g, real_l)
            l_rec = rec_loss(fake_thermal, thermal)
            g_loss = cfg.LAMBDA_ADV * l_adv + cfg.LAMBDA_REC * l_rec
            # Physics loss is ZERO here — ablation!
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.GRAD_CLIP)
            opt_G.step()

            epoch_d += d_loss.item()
            epoch_g += g_loss.item()

        sched_G.step()
        sched_D.step()
        elapsed = time.time() - t0
        n = len(loader)
        print(f"[ABLATION] Epoch [{epoch+1:3d}/{cfg.EPOCHS}]  "
              f"D: {epoch_d/n:.4f}  G: {epoch_g/n:.4f}  "
              f"Time: {elapsed:.1f}s")

        if (epoch + 1) % 10 == 0:
            G.eval()
            with torch.no_grad():
                rgb_s, therm_s = next(iter(loader))
                rgb_s = rgb_s[:4].to(device)
                therm_s = therm_s[:4].to(device)
                fake_s = G(rgb_s)
                grid = torch.cat([fake_s, therm_s], dim=0)
                save_image(grid * 0.5 + 0.5,
                           os.path.join(save_dir, f'epoch_{epoch+1:04d}.png'), nrow=4)
                ssim_v = compute_ssim(fake_s, therm_s)
                psnr_v = compute_psnr(fake_s, therm_s)
                print(f"  SSIM: {ssim_v:.4f}  PSNR: {psnr_v:.2f} dB")
            G.train()

            torch.save({'G_state': G.state_dict(), 'D_state': D.state_dict()},
                       os.path.join(ckpt_dir, f'ckpt_epoch_{epoch+1:04d}.pth'))
            print(f"  Checkpoint saved")

    print("\nAblation Training complete!")


if __name__ == '__main__':
    main()
