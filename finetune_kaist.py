"""
finetune_kaist.py — Fine-tune the KAIST ODE-UNet from best.pth
===============================================================
Resumes from the existing best checkpoint and continues training
with fine-tuning settings optimised for squeezing more SSIM/PSNR:

  • Generator LR:       5e-5  (10x lower than original 2e-4)
  • Discriminator LR:  2.5e-5 (10x lower)
  • Extra random augmentations: horizontal flip, brightness/contrast jitter
  • CosineAnnealingWarmRestarts (T_0=25) for gentle LR cycling
  • 50 fine-tuning epochs
  • Perceptual sharpening: slightly higher spectral loss weight (0.10)
  • Saves to checkpoints_unet/kaist/  — overwrites best.pth if improved

Usage:
    python finetune_kaist.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image

from models.generator import ODEUNetGenerator
from models.discriminator import LiquidDiscriminator
from losses.physics_loss import PhysicsLoss
from losses.spectral_loss import SpectralLoss
from utils.dataset import ThermalDataset
from utils.metrics import compute_ssim, compute_psnr
from config import Config

# ──────────────────────────────────────────────────────────────────────────────
# Fine-tune hyperparams
# ──────────────────────────────────────────────────────────────────────────────
FINETUNE_EPOCHS = 50
LR_G_FT         = 5e-5       # 10× lower than original
LR_D_FT         = 2.5e-5
LAMBDA_SPEC_FT  = 0.10       # slightly stronger spectral alignment
CKPT_PATH       = './checkpoints_unet/kaist/best.pth'
CKPT_DIR        = './checkpoints_unet/kaist'
RESULTS_DIR     = './results_unet/kaist_finetune'

os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Augmented dataset (stronger than training)
# ──────────────────────────────────────────────────────────────────────────────

class AugmentedThermalDataset(torch.utils.data.Dataset):
    """Wraps ThermalDataset and applies extra augmentations for fine-tuning."""

    def __init__(self, base_dataset):
        self.base = base_dataset
        self.aug_rgb = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            T.RandomRotation(degrees=5),
        ])
        self.aug_both = T.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        rgb, thermal = self.base[idx]
        # Apply same flip to both using a manual seed trick
        if torch.rand(1).item() > 0.5:
            rgb     = T.functional.hflip(rgb)
            thermal = T.functional.hflip(thermal)
        if torch.rand(1).item() > 0.5:
            rgb     = T.functional.vflip(rgb)
            thermal = T.functional.vflip(thermal)
        # Color jitter only on RGB
        rgb_pil = T.ToPILImage()(rgb * 0.5 + 0.5)
        rgb_pil = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)(rgb_pil)
        rgb = T.ToTensor()(rgb_pil) * 2 - 1  # back to [-1, 1]
        return rgb, thermal


def add_instance_noise(x, sigma):
    if sigma <= 0:
        return x
    return x + sigma * torch.randn_like(x)


def main():
    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*65}")
    print(f"  PC-LiquidGAN — KAIST Fine-Tuning Run")
    print(f"  Resuming from : {CKPT_PATH}")
    print(f"  LR_G          : {LR_G_FT}  (was 2e-4)")
    print(f"  LR_D          : {LR_D_FT}  (was 1e-4)")
    print(f"  Extra epochs  : {FINETUNE_EPOCHS}")
    print(f"  Spectral λ    : {LAMBDA_SPEC_FT}  (was 0.05)")
    print(f"{'='*65}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    base_path  = os.path.join(cfg.DATA_DIR, 'kaist')
    train_path = os.path.join(base_path, 'train')
    data_path  = train_path if os.path.isdir(train_path) else base_path

    base_ds = ThermalDataset(
        rgb_dir=os.path.join(data_path, 'rgb'),
        thermal_dir=os.path.join(data_path, 'thermal'),
        img_size=cfg.IMG_SIZE,
    )
    aug_ds = AugmentedThermalDataset(base_ds)
    loader = DataLoader(aug_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                        num_workers=cfg.NUM_WORKERS,
                        pin_memory=(device.type == 'cuda'))
    print(f"Dataset: {len(aug_ds)} samples | Batches/epoch: {len(loader)}\n")

    # ── Models ────────────────────────────────────────────────────────────────
    G = ODEUNetGenerator(input_channels=3, output_channels=1, ode_method='euler').to(device)
    D = LiquidDiscriminator(hidden_size=cfg.HIDDEN_SIZE).to(device)

    # Resume from checkpoint
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    G.load_state_dict(ckpt['G_state'])
    D.load_state_dict(ckpt['D_state'])
    best_ssim = ckpt.get('best_ssim', 0.0)
    print(f"Loaded checkpoint — previous best SSIM: {best_ssim:.4f}\n")

    # ── Optimisers with warm restarts ─────────────────────────────────────────
    opt_G = optim.Adam(G.parameters(), lr=LR_G_FT, betas=(cfg.BETA1, cfg.BETA2))
    opt_D = optim.Adam(D.parameters(), lr=LR_D_FT, betas=(cfg.BETA1, cfg.BETA2))
    sched_G = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_G, T_0=25, T_mult=1)
    sched_D = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_D, T_0=25, T_mult=1)

    # ── Loss functions ────────────────────────────────────────────────────────
    l1_loss     = nn.L1Loss()
    adv_loss    = nn.BCEWithLogitsLoss()
    physics_fn  = PhysicsLoss(alpha=0.001).to(device)
    spectral_fn = SpectralLoss(img_size=cfg.IMG_SIZE, sigma=32.0).to(device)

    # ── Fine-tuning loop ──────────────────────────────────────────────────────
    for epoch in range(FINETUNE_EPOCHS):
        G.train(); D.train()
        t0 = time.time()
        ed, eg, el1, ep, es = 0., 0., 0., 0., 0.

        # Very light noise (model is already trained — no warmup needed)
        noise_sigma = 0.03 * max(0, 1.0 - epoch / 25.0)

        for rgb, thermal in loader:
            rgb     = rgb.to(device)
            thermal = thermal.to(device)
            B       = rgb.size(0)

            # Discriminator update
            opt_D.zero_grad()
            fake_t      = G(rgb).detach()
            real_noisy  = add_instance_noise(thermal, noise_sigma)
            fake_noisy  = add_instance_noise(fake_t,  noise_sigma)
            real_labels = torch.full((B, 1), 0.9,  device=device)
            fake_labels = torch.full((B, 1), 0.05, device=device)
            d_loss = 0.5 * (adv_loss(D(real_noisy), real_labels) +
                            adv_loss(D(fake_noisy), fake_labels))
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.GRAD_CLIP)
            opt_D.step()
            ed += d_loss.item()

            # Generator update (2 per D step)
            for _ in range(2):
                opt_G.zero_grad()
                fake_t = G(rgb)
                l1     = l1_loss(fake_t, thermal)
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
                          LAMBDA_SPEC_FT * l_spec)
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.GRAD_CLIP)
                opt_G.step()
                eg  += g_loss.item()
                el1 += l1.item()
                ep  += l_phys.item()
                es  += l_spec.item()

        sched_G.step()
        sched_D.step()

        n = len(loader)
        elapsed = time.time() - t0
        print(f"[FINETUNE] Ep [{epoch+1:3d}/{FINETUNE_EPOCHS}]  "
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
                           os.path.join(RESULTS_DIR, f'ft_epoch_{epoch+1:04d}.png'),
                           nrow=4)
                ssim_v = compute_ssim(fake_s, therm_s)
                psnr_v = compute_psnr(fake_s, therm_s)
                print(f"  → SSIM: {ssim_v:.4f}  PSNR: {psnr_v:.2f} dB", end='')
                if ssim_v > best_ssim:
                    best_ssim = ssim_v
                    torch.save({
                        'G_state': G.state_dict(),
                        'D_state': D.state_dict(),
                        'epoch': f'finetune_{epoch+1}',
                        'best_ssim': best_ssim
                    }, os.path.join(CKPT_DIR, 'best.pth'))
                    print(f"  ← New best! Saved to {CKPT_DIR}/best.pth", end='')
            torch.save({
                'G_state': G.state_dict(),
                'epoch': f'finetune_{epoch+1}'
            }, os.path.join(CKPT_DIR, f'ft_ckpt_epoch_{epoch+1:04d}.pth'))
            G.train()
            print()

    print(f"\nFine-tuning complete! Best SSIM: {best_ssim:.4f}")


if __name__ == '__main__':
    main()
