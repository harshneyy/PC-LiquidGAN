"""
make_proper_grid.py
====================
Generates a qualitative comparison grid where EVERY ROW is the SAME scene:
  Col 1: Input RGB image (test sample)
  Col 2: Our Generated Thermal (PC-LiquidGAN / ODE-UNet inference)
  Col 3: Ground Truth Thermal

One row per dataset: Medical, CBSR, Agri-Tomato, Agri-Chilli, KAIST
Output: qualitative_grid_fixed.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torchvision.transforms as T
import cv2
import os

# ── Config ───────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, '/home/harshney/Desktop/PC-LiquidGAN')
from models.generator import ODEUNetGenerator
from config import Config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = Config()

# Dataset → (rgb_dir, thermal_dir, checkpoint, sample_idx)
DOMAINS = [
    {
        'name':     'Medical\n(Knee IR)',
        'rgb':      'data/medical/val/rgb',
        'thermal':  'data/medical/val/thermal',
        'ckpt':     'checkpoints_unet/medical/best.pth',
        'idx':      0,
        'ssim':     '0.9631',
        'psnr':     '41.22 dB',
    },
    {
        'name':     'CBSR\n(NIR Face)',
        'rgb':      'data/cbsr/val/rgb',
        'thermal':  'data/cbsr/val/thermal',
        'ckpt':     'checkpoints_unet/cbsr/best.pth',
        'idx':      0,
        'ssim':     '0.9976',
        'psnr':     '52.23 dB',
    },
    {
        'name':     'Agri\n(Tomato Leaf)',
        'rgb':      'data/agri/val/rgb',
        'thermal':  'data/agri/val/thermal',
        'ckpt':     'checkpoints_unet/agri/best.pth',
        'idx':      0,
        'ssim':     '0.9945',
        'psnr':     '50.30 dB',
    },
    {
        'name':     'Agri\n(Chilli Leaf)',
        'rgb':      'data/chilli/val/rgb',
        'thermal':  'data/chilli/val/thermal',
        'ckpt':     'checkpoints_unet/chilli/best.pth',
        'idx':      0,
        'ssim':     '0.9947',
        'psnr':     '50.84 dB',
    },
    {
        'name':     'KAIST\n(Surveillance)',
        'rgb':      'data/kaist/val/rgb',
        'thermal':  'data/kaist/val/thermal',
        'ckpt':     'checkpoints_unet/kaist/best.pth',
        'idx':      0,
        'ssim':     '0.9351',
        'psnr':     '37.87 dB',
    },
]

# ── Transforms ───────────────────────────────────────────────────────────────
rgb_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

gt_transform = T.Compose([
    T.Resize((256, 256)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_model(ckpt_path):
    G = ODEUNetGenerator(input_channels=3, output_channels=1, ode_method='euler').to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    G.load_state_dict(state['G_state'])
    G.eval()
    return G

def tensor_to_gray(t):
    t = t.squeeze().cpu().detach()
    t = (t * 0.5 + 0.5).clamp(0, 1)
    return (t.numpy() * 255).astype(np.uint8)

def contrast_stretch(gray):
    mn, mx = gray.min(), gray.max()
    if mx - mn < 1:
        return gray
    return ((gray.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)

def get_sorted_files(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
    return files

# ── Build figure ──────────────────────────────────────────────────────────────
n_rows = len(DOMAINS)
n_cols = 3

fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 2.5))
fig.patch.set_facecolor('#0d0d0d')

COL_TITLES = ['Input RGB Image', 'Our Generated Thermal\n(PC-LiquidGAN)', 'Ground Truth Thermal']
COL_COLORS = ['#4fc3f7', '#ff8a65', '#81c784']

for col_i, (title, color) in enumerate(zip(COL_TITLES, COL_COLORS)):
    axes[0, col_i].set_title(title, color=color, fontsize=11, fontweight='bold', pad=8)

for row_i, domain in enumerate(DOMAINS):
    print(f"Processing {domain['name'].replace(chr(10),' ')} ...")

    # ── 1. Load files (make sure SAME index → same scene) ─────────────────
    rgb_files     = get_sorted_files(domain['rgb'])
    thermal_files = get_sorted_files(domain['thermal'])

    # Match by filename stem if possible, otherwise use index
    idx = domain['idx']
    rgb_file     = rgb_files[idx]
    rgb_stem     = os.path.splitext(rgb_file)[0]

    # Find the matching thermal file with the same stem
    matching_thermal = None
    for tf in thermal_files:
        if os.path.splitext(tf)[0] == rgb_stem:
            matching_thermal = tf
            break
    if matching_thermal is None:
        matching_thermal = thermal_files[idx]  # fallback to same index

    rgb_path     = os.path.join(domain['rgb'],     rgb_file)
    thermal_path = os.path.join(domain['thermal'], matching_thermal)

    rgb_pil = Image.open(rgb_path).convert('RGB')
    gt_pil  = Image.open(thermal_path).convert('L')

    # ── 2. Model inference ────────────────────────────────────────────────
    G = load_model(domain['ckpt'])
    rgb_t = rgb_transform(rgb_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        fake = G(rgb_t)
    gen_gray = tensor_to_gray(fake)
    gen_gray = contrast_stretch(gen_gray)

    gt_gray  = np.array(gt_pil.resize((256, 256)))

    # ── 3. Plot ───────────────────────────────────────────────────────────
    ax_rgb = axes[row_i, 0]
    ax_gen = axes[row_i, 1]
    ax_gt  = axes[row_i, 2]

    rgb_display = np.array(rgb_pil.resize((256, 256)))
    ax_rgb.imshow(rgb_display)
    ax_gen.imshow(gen_gray, cmap='gray', vmin=0, vmax=255)
    ax_gt.imshow(gt_gray,  cmap='gray', vmin=0, vmax=255)

    # Row label (dataset name on left)
    ax_rgb.set_ylabel(domain['name'], color='white', fontsize=9,
                      fontweight='bold', rotation=0, labelpad=60,
                      va='center')

    # SSIM/PSNR badge on generated column
    ax_gen.set_xlabel(f"SSIM {domain['ssim']}  PSNR {domain['psnr']}",
                      color='#ff8a65', fontsize=7.5, labelpad=3)

    for ax in [ax_rgb, ax_gen, ax_gt]:
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        ax.set_facecolor('#0d0d0d')

plt.suptitle('PC-LiquidGAN — Qualitative Comparison (Same Test Sample per Row)',
             color='white', fontsize=13, fontweight='bold', y=1.01)

plt.tight_layout(h_pad=0.4, w_pad=0.3)
out_path = '/home/harshney/Desktop/PC-LiquidGAN/qualitative_grid_fixed.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"\n✅ Saved → {out_path}")
