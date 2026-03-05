"""
train_baseline.py
Train DCGAN or WGAN-GP baselines on thermal datasets for comparative evaluation.

Usage:
    python train_baseline.py --model dcgan --dataset agri --epochs 100
    python train_baseline.py --model wgan-gp --dataset agri --epochs 100
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.dcgan import DCGANGenerator, DCGANDiscriminator
from models.wgan_gp import WGANGenerator, WGANCritic, compute_gradient_penalty
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


def get_args():
    p = argparse.ArgumentParser(description='Baseline GAN Training')
    p.add_argument('--model',      type=str, required=True, choices=['dcgan', 'wgan-gp'])
    p.add_argument('--dataset',    type=str, default='agri',
                   choices=['kaist', 'cbsr', 'medical', 'agri', 'chilli'])
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    return p.parse_args()


def train_dcgan(G, D, loader, cfg, device, args):
    """Standard DCGAN training loop with BCE adversarial + L1 reconstruction loss."""
    opt_G = optim.Adam(G.parameters(), lr=cfg.LR_G, betas=(cfg.BETA1, cfg.BETA2))
    opt_D = optim.Adam(D.parameters(), lr=cfg.LR_D, betas=(cfg.BETA1, cfg.BETA2))
    criterion = nn.BCELoss()
    l1_loss = nn.L1Loss()

    save_dir = f'./results_baseline/dcgan_{args.dataset}'
    ckpt_dir = f'./checkpoints_baseline/dcgan_{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_d, epoch_g = 0.0, 0.0

        for rgb, real_thermal in loader:
            rgb = rgb.to(device)
            real_thermal = real_thermal.to(device)
            B = rgb.size(0)

            # ── Train Discriminator ──
            opt_D.zero_grad()
            fake_thermal = G(rgb).detach()
            d_real = D(rgb, real_thermal)
            d_fake = D(rgb, fake_thermal)
            d_loss = criterion(d_real, torch.ones_like(d_real)) + \
                     criterion(d_fake, torch.zeros_like(d_fake))
            d_loss.backward()
            opt_D.step()

            # ── Train Generator ──
            opt_G.zero_grad()
            fake_thermal = G(rgb)
            d_fake = D(rgb, fake_thermal)
            g_adv = criterion(d_fake, torch.ones_like(d_fake))
            g_rec = l1_loss(fake_thermal, real_thermal) * cfg.LAMBDA_REC
            g_loss = g_adv + g_rec
            g_loss.backward()
            opt_G.step()

            epoch_d += d_loss.item()
            epoch_g += g_loss.item()

        elapsed = time.time() - t0
        n = len(loader)
        print(f"[DCGAN] Epoch [{epoch+1:3d}/{args.epochs}]  "
              f"D: {epoch_d/n:.4f}  G: {epoch_g/n:.4f}  Time: {elapsed:.1f}s")

        # Evaluate every 10 epochs
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

            # Save checkpoint
            torch.save({'G_state': G.state_dict(), 'D_state': D.state_dict()},
                       os.path.join(ckpt_dir, f'ckpt_epoch_{epoch+1:04d}.pth'))
            print(f"  Checkpoint saved")

    print("\nDCGAN Training complete!")


def train_wgan_gp(G, C, loader, cfg, device, args):
    """WGAN-GP training loop with Wasserstein loss + gradient penalty + L1."""
    opt_G = optim.Adam(G.parameters(), lr=cfg.LR_G, betas=(0.0, 0.9))
    opt_C = optim.Adam(C.parameters(), lr=cfg.LR_D, betas=(0.0, 0.9))
    l1_loss = nn.L1Loss()
    n_critic = 5  # Train critic N times per generator step
    lambda_gp = 10.0

    save_dir = f'./results_baseline/wgan_{args.dataset}'
    ckpt_dir = f'./checkpoints_baseline/wgan_{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_c, epoch_g = 0.0, 0.0

        for i, (rgb, real_thermal) in enumerate(loader):
            rgb = rgb.to(device)
            real_thermal = real_thermal.to(device)

            # ── Train Critic (n_critic times) ──
            opt_C.zero_grad()
            fake_thermal = G(rgb).detach()

            c_real = C(rgb, real_thermal).mean()
            c_fake = C(rgb, fake_thermal).mean()
            gp = compute_gradient_penalty(C, rgb, real_thermal, fake_thermal, device)

            c_loss = c_fake - c_real + lambda_gp * gp
            c_loss.backward()
            opt_C.step()
            epoch_c += c_loss.item()

            # ── Train Generator (every n_critic steps) ──
            if i % n_critic == 0:
                opt_G.zero_grad()
                fake_thermal = G(rgb)
                g_adv = -C(rgb, fake_thermal).mean()
                g_rec = l1_loss(fake_thermal, real_thermal) * cfg.LAMBDA_REC
                g_loss = g_adv + g_rec
                g_loss.backward()
                opt_G.step()
                epoch_g += g_loss.item()

        elapsed = time.time() - t0
        n = len(loader)
        n_g = max(n // n_critic, 1)
        print(f"[WGAN-GP] Epoch [{epoch+1:3d}/{args.epochs}]  "
              f"C: {epoch_c/n:.4f}  G: {epoch_g/n_g:.4f}  Time: {elapsed:.1f}s")

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

            torch.save({'G_state': G.state_dict(), 'C_state': C.state_dict()},
                       os.path.join(ckpt_dir, f'ckpt_epoch_{epoch+1:04d}.pth'))
            print(f"  Checkpoint saved")

    print("\nWGAN-GP Training complete!")


def main():
    args = get_args()
    cfg = Config()
    cfg.BATCH_SIZE = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  Baseline Training: {args.model.upper()}")
    print(f"  Dataset: {args.dataset.upper()}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Load dataset
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

    if args.model == 'dcgan':
        G = DCGANGenerator().to(device)
        D = DCGANDiscriminator().to(device)
        G.apply(weights_init)
        D.apply(weights_init)
        print(f"DCGAN Generator params  : {sum(p.numel() for p in G.parameters()):,}")
        print(f"DCGAN Discriminator params: {sum(p.numel() for p in D.parameters()):,}\n")
        train_dcgan(G, D, loader, cfg, device, args)

    elif args.model == 'wgan-gp':
        G = WGANGenerator().to(device)
        C = WGANCritic().to(device)
        G.apply(weights_init)
        C.apply(weights_init)
        print(f"WGAN Generator params: {sum(p.numel() for p in G.parameters()):,}")
        print(f"WGAN Critic params   : {sum(p.numel() for p in C.parameters()):,}\n")
        train_wgan_gp(G, C, loader, cfg, device, args)


if __name__ == '__main__':
    main()
