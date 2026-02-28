"""
report_metrics.py
Generates a clean tabular metrics summary for your final report.

Computes SSIM, PSNR, RMSE on the dataset and saves:
  - results/metrics_table.png    (ready to paste in report)
  - results/metrics_summary.txt  (raw numbers)

Run:
    python report_metrics.py --checkpoint checkpoints/ckpt_epoch_0050.pth
    python report_metrics.py --all_checkpoints     # compare all saved checkpoints
"""

import os
import glob
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.generator import NeuralODEGenerator
from utils.dataset    import ThermalDataset, SyntheticThermalDataset
from utils.metrics    import compute_ssim, compute_psnr, compute_rmse
from config           import Config


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',        type=str, default=None)
    p.add_argument('--dataset',           type=str, default='synthetic',
                   choices=['synthetic', 'kaist', 'cbsr', 'neonatal'])
    p.add_argument('--all_checkpoints',   action='store_true',
                   help='Evaluate all saved checkpoints to see progression')
    p.add_argument('--num_batches',       type=int, default=20,
                   help='Batches to evaluate (default 20 = ~80 images)')
    return p.parse_args()


def evaluate_checkpoint(ckpt_path, G, loader, device, num_batches):
    """Evaluate a checkpoint and return dict of metrics."""
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(ckpt['G_state'])
        epoch = ckpt.get('epoch', 0) + 1
    else:
        epoch = 0

    G.eval()
    ssim_scores, psnr_scores, rmse_scores = [], [], []

    with torch.no_grad():
        for i, (rgb, thermal) in enumerate(loader):
            if i >= num_batches:
                break
            rgb = rgb.to(device)
            thermal = thermal.to(device)
            fake = G(rgb)
            ssim_scores.append(compute_ssim(fake, thermal))
            psnr_scores.append(compute_psnr(fake, thermal))
            rmse_scores.append(compute_rmse(fake, thermal))

    return {
        'epoch': epoch,
        'ssim':  np.mean(ssim_scores),
        'psnr':  np.mean(psnr_scores),
        'rmse':  np.mean(rmse_scores),
        'ssim_std': np.std(ssim_scores),
        'psnr_std': np.std(psnr_scores),
    }


def save_metrics_table(results, save_path='results/metrics_table.png'):
    """Render results as a publication-quality table image."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 1 + len(results) * 0.5))
    ax.axis('off')

    col_labels = ['Epoch', 'SSIM (↑)', 'PSNR dB (↑)', 'RMSE (↓)']
    table_data = [
        [str(r['epoch']),
         f"{r['ssim']:.4f} ± {r['ssim_std']:.4f}",
         f"{r['psnr']:.2f} ± {r['psnr_std']:.2f}",
         f"{r['rmse']:.4f}"]
        for r in results
    ]

    table = ax.table(
        cellText   = table_data,
        colLabels  = col_labels,
        loc        = 'center',
        cellLoc    = 'center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#1f6feb')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Zebra stripes
    for i in range(len(results)):
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor('#f0f6ff' if i % 2 == 0 else '#ffffff')

    # Highlight best SSIM row
    best_idx = np.argmax([r['ssim'] for r in results])
    for j in range(len(col_labels)):
        table[best_idx+1, j].set_facecolor('#e6ffed')
        table[best_idx+1, j].set_text_props(fontweight='bold')

    ax.set_title('PC-LiquidGAN — Evaluation Metrics', fontsize=13,
                 fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Metrics table saved: {save_path}")


def save_metrics_curves(results, save_path='results/metrics_curves.png'):
    """Plot SSIM and PSNR vs epoch curves."""
    if len(results) < 2:
        return  # Need at least 2 points for a curve

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = [r['epoch'] for r in results]
    ssims  = [r['ssim']  for r in results]
    psnrs  = [r['psnr']  for r in results]
    rmses  = [r['rmse']  for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('PC-LiquidGAN Training Progress', fontsize=13, fontweight='bold')

    for ax, ys, label, color, marker in zip(
        axes,
        [ssims, psnrs, rmses],
        ['SSIM (higher = better)', 'PSNR dB (higher = better)', 'RMSE (lower = better)'],
        ['#1f6feb', '#1a7f37', '#b91c1c'],
        ['o', 's', '^']
    ):
        ax.plot(epochs, ys, color=color, marker=marker, linewidth=2, markersize=6)
        ax.fill_between(epochs, ys, alpha=0.1, color=color)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Metrics curves saved: {save_path}")


def main():
    args   = get_args()
    cfg    = Config()
    device = torch.device('cpu')

    # Dataset
    if args.dataset == 'synthetic':
        dataset = SyntheticThermalDataset(num_samples=200, img_size=256)
    else:
        base       = f'./data/{args.dataset}'
        train_path = os.path.join(base, 'train')
        data_path  = train_path if os.path.isdir(train_path) else base
        dataset    = ThermalDataset(
            rgb_dir     = os.path.join(data_path, 'rgb'),
            thermal_dir = os.path.join(data_path, 'thermal'),
            img_size=256, augment=False,
        )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Generator
    G = NeuralODEGenerator(
        latent_dim=cfg.LATENT_DIM, ode_method='euler', rtol=1e-2, atol=1e-2
    ).to(device)

    os.makedirs('./results', exist_ok=True)

    if args.all_checkpoints:
        ckpts = sorted(glob.glob('./checkpoints/*.pth'))
        if not ckpts:
            print("[WARN] No checkpoints found. Train first: python train.py --dataset synthetic --epochs 50")
            return
        all_results = []
        for ckpt in ckpts:
            print(f"Evaluating: {ckpt}")
            r = evaluate_checkpoint(ckpt, G, loader, device, args.num_batches)
            all_results.append(r)
            print(f"  Epoch {r['epoch']:3d} | SSIM={r['ssim']:.4f} | PSNR={r['psnr']:.2f} dB | RMSE={r['rmse']:.4f}")
        save_metrics_table(all_results)
        save_metrics_curves(all_results)
    else:
        ckpt = args.checkpoint or sorted(glob.glob('./checkpoints/*.pth') or [''])[-1]
        print(f"Evaluating: {ckpt or '(untrained)'}")
        r = evaluate_checkpoint(ckpt if ckpt else None, G, loader, device, args.num_batches)
        print(f"\nResults: SSIM={r['ssim']:.4f} | PSNR={r['psnr']:.2f} dB | RMSE={r['rmse']:.4f}")
        save_metrics_table([r])

    # Save text summary
    txt_path = 'results/metrics_summary.txt'
    with open(txt_path, 'w') as f:
        f.write("PC-LiquidGAN Evaluation Metrics\n")
        f.write("="*50 + "\n")
        f.write(f"Dataset: {args.dataset}\n\n")
        f.write(f"{'Epoch':>8} {'SSIM':>10} {'PSNR (dB)':>12} {'RMSE':>10}\n")
        f.write("-"*50 + "\n")
    print(f"[OK] Text summary saved: {txt_path}")


if __name__ == '__main__':
    main()
