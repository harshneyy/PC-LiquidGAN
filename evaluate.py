"""
evaluate.py
Evaluation script for PC-LiquidGAN.

Run:
    python evaluate.py --checkpoint checkpoints/ckpt_epoch_0100.pth --dataset kaist
    python evaluate.py --checkpoint checkpoints/ckpt_epoch_0100.pth --dataset cbsr  (zero-shot)
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.generator  import NeuralODEGenerator
from utils.dataset     import ThermalDataset, UnpairedDataset
from utils.metrics     import compute_ssim, compute_psnr, compute_rmse
from config            import Config


def get_args():
    p = argparse.ArgumentParser(description='PC-LiquidGAN Evaluation')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to saved checkpoint .pth file')
    p.add_argument('--dataset',    type=str, default='kaist',
                   choices=['kaist', 'cbsr', 'neonatal'],
                   help='Dataset to evaluate on (for zero-shot: pick a different domain)')
    p.add_argument('--save_images', action='store_true',
                   help='Save generated thermal images to results/')
    p.add_argument('--unpaired', action='store_true',
                   help='Run in zero-shot mode (no GT thermal, only RGB input)')
    return p.parse_args()


def evaluate():
    args   = get_args()
    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  PC-LiquidGAN Evaluation")
    print(f"  Device    : {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Mode      : {'Zero-Shot (Unpaired)' if args.unpaired else 'Supervised (Paired)'}")
    print(f"{'='*60}\n")

    # ── Load Model ────────────────────────────────────────────────────────────
    G = NeuralODEGenerator(latent_dim=cfg.LATENT_DIM).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(ckpt['G_state'])
    G.eval()
    print(f"Model loaded from epoch {ckpt.get('epoch', '?')}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_path = os.path.join(cfg.DATA_DIR, args.dataset)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    if args.unpaired:
        dataset = UnpairedDataset(
            img_dir  = os.path.join(data_path, 'rgb'),
            img_size = cfg.IMG_SIZE,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        print("Running zero-shot inference (no GT thermal)...")
        with torch.no_grad():
            for i, (rgb, fname) in enumerate(loader):
                rgb  = rgb.to(device)
                fake = G(rgb)
                if args.save_images:
                    out_path = os.path.join(cfg.RESULTS_DIR, f'zeroshot_{fname[0]}')
                    save_image(fake * 0.5 + 0.5, out_path)
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(loader)} images")
        print(f"\n✅ Zero-shot inference complete. Images saved to {cfg.RESULTS_DIR}")
        return

    # Paired evaluation
    dataset = ThermalDataset(
        rgb_dir     = os.path.join(data_path, 'rgb'),
        thermal_dir = os.path.join(data_path, 'thermal'),
        img_size    = cfg.IMG_SIZE,
        augment     = False,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False,
                        num_workers=cfg.NUM_WORKERS)

    all_ssim, all_psnr, all_rmse = [], [], []

    with torch.no_grad():
        for i, (rgb, thermal) in enumerate(loader):
            rgb     = rgb.to(device)
            thermal = thermal.to(device)
            fake    = G(rgb)

            all_ssim.append(compute_ssim(fake, thermal))
            all_psnr.append(compute_psnr(fake, thermal))
            all_rmse.append(compute_rmse(fake, thermal))

            if args.save_images and i < 10:  # Save first 10 batches
                grid = torch.cat([fake, thermal], dim=0)
                save_image(
                    grid * 0.5 + 0.5,
                    os.path.join(cfg.RESULTS_DIR, f'eval_batch_{i:04d}.png'),
                    nrow=4
                )

            if (i + 1) % 20 == 0:
                print(f"  Batch {i+1}/{len(loader)}")

    # ── Print Results ─────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Evaluation Results on {args.dataset.upper()}")
    print(f"{'='*50}")
    print(f"  SSIM (↑)  : {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    print(f"  PSNR (↑)  : {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
    print(f"  RMSE (↓)  : {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    evaluate()
