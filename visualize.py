"""
visualize.py
Generates side-by-side visual comparisons for the final report.

Creates:
  results/comparison_grid.png  — RGB | Generated Thermal | Real Thermal
  results/loss_curves.png      — Training loss plots from TensorBoard logs

Run:
    python visualize.py                                    # use latest checkpoint
    python visualize.py --checkpoint checkpoints/ckpt_epoch_0050.pth
    python visualize.py --no_checkpoint                    # random model (untrained demo)
"""

import os
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import torch
from torchvision.utils import make_grid
from PIL import Image

from models.generator  import NeuralODEGenerator
from utils.dataset     import ThermalDataset, SyntheticThermalDataset
from torch.utils.data  import DataLoader
from config            import Config


# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',    type=str,  default=None)
    p.add_argument('--dataset',       type=str,  default='synthetic',
                   choices=['synthetic', 'kaist', 'cbsr', 'neonatal'])
    p.add_argument('--num_samples',   type=int,  default=8,  help='Images to show in grid')
    p.add_argument('--no_checkpoint', action='store_true',    help='Use untrained model')
    return p.parse_args()


def find_latest_checkpoint(ckpt_dir='./checkpoints'):
    """Auto-find the latest checkpoint file."""
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth')))
    return ckpts[-1] if ckpts else None


def load_generator(checkpoint_path, cfg, device):
    G = NeuralODEGenerator(
        latent_dim = cfg.LATENT_DIM,
        ode_method = 'euler',   # fast for inference
        rtol=1e-2, atol=1e-2,
    ).to(device)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        G.load_state_dict(ckpt['G_state'])
        epoch = ckpt.get('epoch', '?')
        print(f"[OK] Loaded checkpoint from epoch {epoch}: {checkpoint_path}")
    else:
        print("[INFO] Using untrained model (no checkpoint found)")
    G.eval()
    return G


def tensor_to_rgb(t):
    """Convert [-1,1] tensor to [0,1] numpy for display."""
    return np.clip((t.cpu().numpy() * 0.5 + 0.5), 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  1. Comparison Grid: RGB | Generated Thermal | Real Thermal
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_grid(G, loader, device, num_samples=8, save_path='results/comparison_grid.png'):
    """Creates a 3-column grid: Input RGB | Generated Thermal | Real Thermal."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rgb_list, fake_list, real_list = [], [], []

    with torch.no_grad():
        for rgb, thermal in loader:
            rgb     = rgb.to(device)
            thermal = thermal.to(device)
            fake    = G(rgb)

            for i in range(min(rgb.size(0), num_samples - len(rgb_list))):
                rgb_list.append(tensor_to_rgb(rgb[i]))          # [3, H, W]
                fake_list.append(tensor_to_rgb(fake[i]))        # [1, H, W]
                real_list.append(tensor_to_rgb(thermal[i]))     # [1, H, W]

            if len(rgb_list) >= num_samples:
                break

    n = len(rgb_list)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    fig.suptitle('PC-LiquidGAN: Thermal Image Synthesis Results',
                 fontsize=16, fontweight='bold', y=1.01)

    col_titles = ['Input (RGB / Visible)', 'Generated Thermal\n(PC-LiquidGAN)', 'Ground Truth Thermal']
    cmaps      = [None, 'inferno', 'inferno']

    for col, (title, cmap) in enumerate(zip(col_titles, cmaps)):
        for row in range(n):
            ax = axes[row, col] if n > 1 else axes[col]

            if col == 0:
                img = np.transpose(rgb_list[row], (1, 2, 0))  # [H, W, 3]
                ax.imshow(img)
            else:
                data = fake_list[row][0] if col == 1 else real_list[row][0]
                ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
                if col == 1:
                    # Add a subtle colorbar for thermal
                    plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

            ax.axis('off')
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Comparison grid saved: {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
#  2. Architecture Diagram (text-based, no graphviz needed)
# ─────────────────────────────────────────────────────────────────────────────

def plot_architecture(save_path='results/architecture.png'):
    """Plots a simple block diagram of PC-LiquidGAN architecture."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    def box(x, y, w, h, color, label, sublabel=''):
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5,
                              edgecolor='white', facecolor=color, alpha=0.85, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
                ha='center', va='center', color='white', fontsize=9,
                fontweight='bold', zorder=3)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.28, sublabel,
                    ha='center', va='center', color='#aaaaaa', fontsize=7, zorder=3)

    def arrow(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=1.5))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.15
            ax.text(mx, my, label, ha='center', va='bottom',
                    color='#58a6ff', fontsize=7.5)

    # Generator path (top)
    box(0.2, 2.8, 1.8, 1.2, '#1f6feb', 'RGB Image',  '3×256×256')
    arrow(2.0, 3.4, 2.4, 3.4)
    box(2.4, 2.8, 2.0, 1.2, '#1a7f37', 'CNN Encoder', '512×4×4')
    arrow(4.4, 3.4, 4.8, 3.4)
    box(4.8, 2.8, 2.0, 1.2, '#9a3ecb', 'Neural ODE\n(dopri5)', 'z₀→z₁  [128]')
    arrow(6.8, 3.4, 7.2, 3.4)
    box(7.2, 2.8, 2.0, 1.2, '#1a7f37', 'CNN Decoder', '1×256×256')
    arrow(9.2, 3.4, 9.6, 3.4)
    box(9.6, 2.8, 1.9, 1.2, '#bf8700', 'Fake Thermal', '1×256×256')

    # Discriminator path (bottom)
    box(9.6, 0.8, 1.9, 1.2, '#bf8700', 'Real/Fake', 'Thermal')
    arrow(11.5, 1.4, 11.9, 1.4)
    box(11.9, 0.8, 1.8, 1.2, '#b91c1c', 'LNN Disc.', 'CNN+LiquidCell')

    # Physics loss
    ax.annotate('', xy=(10.5, 2.8), xytext=(10.5, 2.0),
                arrowprops=dict(arrowstyle='->', color='#f78166', lw=1.5))
    box(9.6, 1.2, 1.9, 0.8, '#842029', 'Physics Loss', 'Heat + Energy')

    # Feedback arrow
    ax.annotate('', xy=(0.5, 2.8), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='#ffffff', lw=1,
                                connectionstyle='arc3,rad=0.0'))
    ax.annotate('', xy=(0.5, 0.5), xytext=(12.8, 0.5),
                arrowprops=dict(arrowstyle='->', color='#ffffff', lw=1))
    ax.annotate('', xy=(12.8, 0.5), xytext=(12.8, 1.4),
                arrowprops=dict(arrowstyle='-', color='#ffffff', lw=1))
    ax.text(6.65, 0.25, 'Backpropagation (Adjoint Method)',
            ha='center', color='#aaaaaa', fontsize=8)

    # Labels
    ax.text(7.0, 4.6, 'PC-LiquidGAN Architecture',
            ha='center', color='white', fontsize=14, fontweight='bold')
    ax.text(6.0, 2.6, 'GENERATOR  (Neural ODE)',
            ha='center', color='#58a6ff', fontsize=9, style='italic')
    ax.text(12.8, 2.6, 'DISC.',
            ha='center', color='#f78166', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"[OK] Architecture diagram saved: {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
#  3. Physics Loss Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_physics_explanation(save_path='results/physics_loss_explain.png'):
    """Visualizes how the Heat Diffusion loss works on a sample thermal image."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    import torch.nn.functional as F

    # Create a synthetic "thermal" image with a hot spot
    H, W = 128, 128
    thermal = np.zeros((H, W), dtype=np.float32)
    # Hot spot in center
    yy, xx = np.mgrid[:H, :W]
    thermal += 0.3 * np.exp(-((xx - 64)**2 + (yy - 64)**2) / (2 * 20**2))
    thermal += 0.15 * np.exp(-((xx - 90)**2 + (yy - 40)**2) / (2 * 15**2))
    thermal += np.random.normal(0, 0.02, (H, W))
    thermal = np.clip(thermal, 0, 1)

    T = torch.tensor(thermal).unsqueeze(0).unsqueeze(0)
    lap_kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]])
    laplacian  = F.conv2d(T, lap_kernel, padding=1).squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('Heat Diffusion Physics Loss: dT/dt = α∇²T',
                 fontsize=13, fontweight='bold')

    im0 = axes[0].imshow(thermal, cmap='inferno', vmin=0, vmax=thermal.max())
    axes[0].set_title('T(x,y) — Temperature Field', fontweight='bold')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(laplacian, cmap='RdBu_r')
    axes[1].set_title('∇²T — Laplacian (Heat Flow)', fontweight='bold')
    plt.colorbar(im1, ax=axes[1])

    residual = np.abs(thermal - 0.001 * laplacian)
    im2 = axes[2].imshow(residual, cmap='hot')
    axes[2].set_title('|dT/dt − α∇²T| — Physics Residual\n(Loss to Minimize)',
                      fontweight='bold')
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Physics explanation saved: {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    cfg    = Config()
    device = torch.device('cpu')   # inference on CPU is fine for visualization

    os.makedirs('./results', exist_ok=True)

    # Find checkpoint
    ckpt_path = None
    if not args.no_checkpoint:
        ckpt_path = args.checkpoint or find_latest_checkpoint()
        if ckpt_path:
            print(f"Using checkpoint: {ckpt_path}")
        else:
            print("No checkpoint found — generating figures with untrained model")

    # Load generator
    G = load_generator(ckpt_path, cfg, device)

    # Load a few samples
    print("\nLoading dataset samples...")
    if args.dataset == 'synthetic':
        dataset = SyntheticThermalDataset(num_samples=32, img_size=256)
    else:
        base = f'./data/{args.dataset}'
        train_path = os.path.join(base, 'train')
        data_path  = train_path if os.path.isdir(train_path) else base
        dataset    = ThermalDataset(
            rgb_dir     = os.path.join(data_path, 'rgb'),
            thermal_dir = os.path.join(data_path, 'thermal'),
            img_size    = 256, augment=False,
        )

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Generate all figures
    print("\nGenerating report figures...")
    make_comparison_grid(G, loader, device,
                         num_samples  = args.num_samples,
                         save_path    = 'results/comparison_grid.png')
    plot_architecture(save_path='results/architecture.png')
    plot_physics_explanation(save_path='results/physics_loss_explain.png')

    print("\n" + "="*55)
    print("  All figures saved to results/")
    print("  - results/comparison_grid.png      (for final report)")
    print("  - results/architecture.png          (for presentation)")
    print("  - results/physics_loss_explain.png  (for methodology)")
    print("="*55)


if __name__ == '__main__':
    main()
