"""
train.py
Main training script for PC-LiquidGAN.

Run:
    python train.py              # Train on KAIST with default settings
    python train.py --epochs 50  # Custom epochs
    python train.py --test       # Quick test with synthetic data (no real data needed)
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data   import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils  import save_image

from models.generator     import NeuralODEGenerator
from models.discriminator import LiquidDiscriminator
from losses.physics_loss  import PhysicsLoss
from utils.dataset        import ThermalDataset, SyntheticThermalDataset
from utils.metrics        import compute_ssim, compute_psnr
from config               import Config


# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='PC-LiquidGAN Training')
    p.add_argument('--epochs',     type=int,   default=None)
    p.add_argument('--batch_size', type=int,   default=None)
    p.add_argument('--lr_g',       type=float, default=None)
    p.add_argument('--lr_d',       type=float, default=None)
    p.add_argument('--img_size',   type=int,   default=None)
    p.add_argument('--dataset',    type=str,   default='synthetic',
                   choices=['kaist', 'cbsr', 'neonatal', 'medical', 'agri', 'chilli', 'synthetic'])
    p.add_argument('--resume',     type=str,   default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--test',       action='store_true',
                   help='Use synthetic data — quick pipeline test')
    return p.parse_args()


def weights_init(m):
    """Apply He initialization to Conv/ConvTranspose layers."""
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


# ─────────────────────────────────────────────────────────────────────────────

def train():
    args   = get_args()
    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  PC-LiquidGAN Training")
    print(f"  Device : {device}")
    print(f"  Mode   : {'SYNTHETIC TEST' if args.test else args.dataset.upper()}")
    print(f"{'='*60}\n")

    # Override config with CLI args
    if args.epochs:     cfg.EPOCHS     = args.epochs
    if args.batch_size: cfg.BATCH_SIZE = args.batch_size
    if args.lr_g:       cfg.LR_G       = args.lr_g
    if args.lr_d:       cfg.LR_D       = args.lr_d
    if args.img_size:   cfg.IMG_SIZE   = args.img_size

    os.makedirs(cfg.SAVE_DIR,    exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # -- Dataset ---------------------------------------------------------------
    if args.test:
        dataset = SyntheticThermalDataset(num_samples=64, img_size=64)
        cfg.IMG_SIZE   = 64
        cfg.BATCH_SIZE = 4
        cfg.EPOCHS     = 3
        print("Using random tensors for quick pipeline test (64x64, 3 epochs)")
    else:
        # Support both flat structure (data/kaist/rgb) and
        # split structure (data/synthetic/train/rgb)
        base_path = os.path.join(cfg.DATA_DIR, args.dataset)
        train_path = os.path.join(base_path, 'train')
        if os.path.isdir(train_path):
            data_path = train_path   # synthetic/train/rgb, synthetic/train/thermal
        else:
            data_path = base_path    # kaist/rgb, kaist/thermal

        dataset = ThermalDataset(
            rgb_dir     = os.path.join(data_path, 'rgb'),
            thermal_dir = os.path.join(data_path, 'thermal'),
            img_size    = cfg.IMG_SIZE,
            augment     = True,
        )


    loader = DataLoader(
        dataset,
        batch_size  = cfg.BATCH_SIZE,
        shuffle     = True,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = (device.type == 'cuda'),
    )
    print(f"Dataset: {len(dataset)} pairs | Batches/epoch: {len(loader)}\n")

    # ── Models ────────────────────────────────────────────────────────────────
    G = NeuralODEGenerator(
        latent_dim  = cfg.LATENT_DIM,
        ode_method  = cfg.ODE_METHOD,
        rtol        = cfg.ODE_RTOL,
        atol        = cfg.ODE_ATOL,
    ).to(device)

    D = LiquidDiscriminator(
        hidden_size = cfg.HIDDEN_SIZE,
    ).to(device)

    G.apply(weights_init)
    D.apply(weights_init)
    print(f"Generator params    : {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}\n")

    # ── Optimizers & Schedulers ───────────────────────────────────────────────
    opt_G = optim.Adam(G.parameters(), lr=cfg.LR_G, betas=(cfg.BETA1, cfg.BETA2))
    opt_D = optim.Adam(D.parameters(), lr=cfg.LR_D, betas=(cfg.BETA1, cfg.BETA2))

    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=cfg.EPOCHS)
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=cfg.EPOCHS)

    # ── Loss Functions ────────────────────────────────────────────────────────
    adv_loss     = nn.BCELoss()
    rec_loss     = nn.L1Loss()
    physics_loss = PhysicsLoss(alpha=cfg.THERMAL_ALPHA).to(device)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt['G_state'])
        D.load_state_dict(ckpt['D_state'])
        opt_G.load_state_dict(ckpt['opt_G'])
        opt_D.load_state_dict(ckpt['opt_D'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    writer = SummaryWriter(cfg.LOG_DIR)
    step   = 0

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.EPOCHS):
        G.train()
        D.train()
        t0 = time.time()
        epoch_d, epoch_g, epoch_p = 0., 0., 0.

        for rgb, thermal in loader:
            rgb     = rgb.to(device)
            thermal = thermal.to(device)
            B       = rgb.size(0)
            real_l  = torch.ones(B, 1,  device=device)
            fake_l  = torch.zeros(B, 1, device=device)

            # ── Discriminator Step ────────────────────────────────────────────
            opt_D.zero_grad()
            fake_thermal = G(rgb).detach()
            d_real = D(thermal)
            d_fake = D(fake_thermal)
            d_loss = 0.5 * (adv_loss(d_real, real_l) + adv_loss(d_fake, fake_l))
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.GRAD_CLIP)
            opt_D.step()

            # ── Generator Step ────────────────────────────────────────────────
            opt_G.zero_grad()
            fake_thermal  = G(rgb)
            d_fake_for_g  = D(fake_thermal)

            l_adv     = adv_loss(d_fake_for_g, real_l)
            l_rec     = rec_loss(fake_thermal, thermal)
            l_physics = physics_loss(
                fake_thermal, thermal,
                lambda_flux    = cfg.LAMBDA_FLUX,
                lambda_energy  = cfg.LAMBDA_ENERGY,
            )
            g_loss = (cfg.LAMBDA_ADV * l_adv +
                      cfg.LAMBDA_REC * l_rec +
                      l_physics)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.GRAD_CLIP)
            opt_G.step()

            epoch_d += d_loss.item()
            epoch_g += g_loss.item()
            epoch_p += l_physics.item()

            # Step-level logging
            if step % cfg.LOG_FREQ == 0:
                writer.add_scalar('Step/D_loss',       d_loss.item(),   step)
                writer.add_scalar('Step/G_loss',       g_loss.item(),   step)
                writer.add_scalar('Step/Physics_loss', l_physics.item(), step)
            step += 1

        # Epoch-level logging
        sched_G.step()
        sched_D.step()
        elapsed = time.time() - t0
        n = len(loader)
        print(f"Epoch [{epoch+1:3d}/{cfg.EPOCHS}]  "
              f"D: {epoch_d/n:.4f}  G: {epoch_g/n:.4f}  "
              f"Physics: {epoch_p/n:.4f}  "
              f"Time: {elapsed:.1f}s")
        writer.add_scalar('Epoch/D_loss',       epoch_d / n, epoch)
        writer.add_scalar('Epoch/G_loss',       epoch_g / n, epoch)
        writer.add_scalar('Epoch/Physics_loss', epoch_p / n, epoch)

        # Save sample images
        if (epoch + 1) % cfg.EVAL_FREQ == 0:
            G.eval()
            with torch.no_grad():
                rgb_s, therm_s = next(iter(loader))
                rgb_s   = rgb_s[:4].to(device)
                therm_s = therm_s[:4].to(device)
                fake_s  = G(rgb_s)
                # Save grid: [fake | real]
                grid = torch.cat([fake_s, therm_s], dim=0)
                save_image(
                    grid * 0.5 + 0.5,
                    os.path.join(cfg.RESULTS_DIR, f'epoch_{epoch+1:04d}.png'),
                    nrow=4
                )
                # Log metrics
                ssim_v = compute_ssim(fake_s, therm_s)
                psnr_v = compute_psnr(fake_s, therm_s)
                writer.add_scalar('Metrics/SSIM', ssim_v, epoch)
                writer.add_scalar('Metrics/PSNR', psnr_v, epoch)
                print(f"  SSIM: {ssim_v:.4f}  PSNR: {psnr_v:.2f} dB")

        # Save checkpoint
        if (epoch + 1) % cfg.SAVE_FREQ == 0:
            torch.save({
                'epoch':   epoch,
                'G_state': G.state_dict(),
                'D_state': D.state_dict(),
                'opt_G':   opt_G.state_dict(),
                'opt_D':   opt_D.state_dict(),
            }, os.path.join(cfg.SAVE_DIR, f'ckpt_epoch_{epoch+1:04d}.pth'))
            print(f"  Checkpoint saved")

    writer.close()
    print("\nTraining complete!")


if __name__ == '__main__':
    train()
