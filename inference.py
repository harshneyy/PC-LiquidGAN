"""
inference.py
Single-image inference demo for PC-LiquidGAN.

Given ONE input image (RGB / visible), generates its thermal equivalent.

Run:
    python inference.py --input path/to/image.jpg
    python inference.py --input path/to/image.jpg --checkpoint checkpoints/ckpt_epoch_0050.pth
    python inference.py --demo                           # runs on a random synthetic image
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path

from models.generator import NeuralODEGenerator
from config           import Config


def get_args():
    p = argparse.ArgumentParser(description='PC-LiquidGAN Single-Image Inference')
    p.add_argument('--input',      type=str, default=None, help='Path to input RGB image')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint .pth')
    p.add_argument('--output',     type=str, default='results/inference_output.png')
    p.add_argument('--img_size',   type=int, default=256)
    p.add_argument('--demo',       action='store_true', help='Run with a random synthetic input')
    return p.parse_args()


def find_latest_checkpoint(ckpt_dir='./checkpoints'):
    import glob
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth')))
    return ckpts[-1] if ckpts else None


def load_image(path, img_size):
    """Load an RGB image and convert to normalized tensor."""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(img).unsqueeze(0)   # [1, 3, H, W]


def generate_thermal(rgb_tensor, G, device):
    """Run the generator and return the thermal output tensor."""
    G.eval()
    with torch.no_grad():
        rgb_tensor = rgb_tensor.to(device)
        fake_thermal = G(rgb_tensor)
    return fake_thermal


def save_output(rgb_tensor, thermal_tensor, output_path):
    """Save a side-by-side comparison image."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    def to_numpy(t, is_rgb=False):
        arr = t.squeeze().cpu().numpy()
        arr = np.clip(arr * 0.5 + 0.5, 0, 1)
        if is_rgb:
            arr = np.transpose(arr, (1, 2, 0))  # [H, W, 3]
        return arr

    rgb_np     = to_numpy(rgb_tensor,     is_rgb=True)
    thermal_np = to_numpy(thermal_tensor, is_rgb=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle('PC-LiquidGAN — Thermal Image Synthesis', fontsize=14, fontweight='bold')

    # Input RGB
    axes[0].imshow(rgb_np)
    axes[0].set_title('Input (RGB / Visible)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Generated thermal (inferno colormap)
    im = axes[1].imshow(thermal_np, cmap='inferno', vmin=0, vmax=1)
    axes[1].set_title('Generated Thermal\n(PC-LiquidGAN)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Temperature (relative)')

    # Thermal with temperature annotations
    axes[2].imshow(thermal_np, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Hot-Zone Visualization', fontsize=12, fontweight='bold')
    # Mark the hottest and coldest points
    hot_y, hot_x = np.unravel_index(thermal_np.argmax(), thermal_np.shape)
    cld_y, cld_x = np.unravel_index(thermal_np.argmin(), thermal_np.shape)
    axes[2].plot(hot_x, hot_y, 'w+', markersize=14, markeredgewidth=2, label='Hottest')
    axes[2].plot(cld_x, cld_y, 'c+', markersize=14, markeredgewidth=2, label='Coldest')
    axes[2].legend(loc='upper right', fontsize=8, framealpha=0.7)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved inference result: {output_path}")

    # Also save the raw thermal as a grayscale PNG
    thermal_uint8 = (thermal_np * 255).astype(np.uint8)
    thermal_img   = Image.fromarray(thermal_uint8, mode='L')
    raw_path      = output_path.replace('.png', '_thermal_raw.png')
    thermal_img.save(raw_path)
    print(f"[OK] Saved raw thermal image: {raw_path}")

    return output_path


def main():
    args   = get_args()
    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nPC-LiquidGAN Inference")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt_path = args.checkpoint or find_latest_checkpoint()
    G = NeuralODEGenerator(
        latent_dim = cfg.LATENT_DIM,
        ode_method = 'euler',     # Fast for inference
        rtol=1e-2, atol=1e-2,
    ).to(device)

    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(ckpt['G_state'])
        print(f"[OK] Model loaded from: {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    else:
        print("[INFO] No checkpoint found — using untrained model (output will be random)")

    # Load input image
    if args.demo or args.input is None:
        print("[DEMO] Using a random synthetic input image...")
        rgb_tensor = torch.randn(1, 3, args.img_size, args.img_size)
    else:
        print(f"[OK] Loading image: {args.input}")
        rgb_tensor = load_image(args.input, args.img_size)

    # Run inference
    print("Running inference...")
    thermal_tensor = generate_thermal(rgb_tensor, G, device)

    # Compute simple stats
    t_np = thermal_tensor.squeeze().cpu().numpy() * 0.5 + 0.5
    print(f"\nThermal image statistics:")
    print(f"  Min temperature : {t_np.min():.4f}  (relative scale)")
    print(f"  Max temperature : {t_np.max():.4f}  (relative scale)")
    print(f"  Mean temperature: {t_np.mean():.4f}  (relative scale)")
    print(f"  Std deviation   : {t_np.std():.4f}")

    # Save output
    save_output(rgb_tensor, thermal_tensor, args.output)


if __name__ == '__main__':
    main()
