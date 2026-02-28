"""
download_datasets.py
Automated dataset downloader for PC-LiquidGAN.

Datasets downloaded:
  1. KAIST Multispectral Preview  (1.44 GB) — paired RGB+Thermal, pedestrian scenes
  2. Synthetic Thermal Dataset     (generated locally, ~50 MB) — instant, no internet
  3. FLIR ADAS sample              (if kaggle credentials available)

Usage:
    python download_datasets.py              # Download all available
    python download_datasets.py --synthetic  # Only generate synthetic data (instant)
    python download_datasets.py --kaist      # Only download KAIST
"""

import os
import sys
import shutil
import argparse
import zipfile
import tarfile
import numpy as np
from pathlib import Path

try:
    from PIL import Image
    print("[OK] PIL available")
except ImportError:
    print("[WARN] PIL not found, installing...")
    os.system("pip install pillow")
    from PIL import Image

DATA_DIR = Path("./data")


# =============================================================================
#  SYNTHETIC DATASET GENERATOR
#  Creates realistic-looking pseudo-thermal images from random patterns.
#  Useful for: testing the pipeline, ablation studies.
# =============================================================================

def generate_synthetic_thermal(rgb_array: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to a pseudo-thermal image.
    Uses a weighted luminance + random thermal noise model:
        T = 0.3*R + 0.59*G + 0.11*B + thermal_noise
    """
    r = rgb_array[:, :, 0].astype(np.float32)
    g = rgb_array[:, :, 1].astype(np.float32)
    b = rgb_array[:, :, 2].astype(np.float32)

    # Luminance as base temperature signal
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    # Simulate heat sources (brighter objects = warmer)
    thermal = luminance + np.random.normal(0, 8, luminance.shape)

    # Apply thermal smoothing (heat diffuses)
    from scipy.ndimage import gaussian_filter
    thermal = gaussian_filter(thermal, sigma=2)

    # Normalize to [0, 255]
    thermal = np.clip(thermal, 0, 255).astype(np.uint8)
    return thermal


def create_synthetic_dataset(
    dataset_name: str = "synthetic",
    num_train: int = 500,
    num_val:   int = 100,
    img_size:  int = 256,
):
    """Generate a synthetic RGB-Thermal paired dataset."""
    print(f"\n[Synthetic Dataset] Generating {num_train} train + {num_val} val pairs ({img_size}x{img_size})...")

    for split, count in [("train", num_train), ("val", num_val)]:
        rgb_dir     = DATA_DIR / dataset_name / split / "rgb"
        thermal_dir = DATA_DIR / dataset_name / split / "thermal"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        thermal_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            # Generate realistic-looking RGB (random scenes with structure)
            # Use gradient + Gaussian blobs for structure
            base = np.random.randint(30, 200, (img_size, img_size, 3), dtype=np.uint8)

            # Add some gradient structure
            from scipy.ndimage import gaussian_filter
            for c in range(3):
                noise = np.random.randn(img_size, img_size) * 50
                base[:, :, c] = np.clip(base[:, :, c] + gaussian_filter(noise, sigma=20), 0, 255).astype(np.uint8)

            rgb_img     = Image.fromarray(base, 'RGB')
            thermal_arr = generate_synthetic_thermal(base)
            thermal_img = Image.fromarray(thermal_arr, 'L')

            fname = f"{i:05d}.png"
            rgb_img.save(rgb_dir / fname)
            thermal_img.save(thermal_dir / fname)

            if (i + 1) % 100 == 0:
                print(f"  [{split}] {i+1}/{count} generated")

    print(f"[OK] Synthetic dataset created at: {DATA_DIR / dataset_name}")
    print(f"     Train: {num_train} pairs | Val: {num_val} pairs")


# =============================================================================
#  KAIST MULTISPECTRAL DATASET
#  gdown ID: 1nJkdnSI9fAuhhZPKFfhVXimXGzvuRTb6  (preview, 1.44 GB)
# =============================================================================

def download_kaist_preview():
    """Download KAIST Multispectral Preview set via gdown."""
    print("\n[KAIST] Downloading preview set (1.44 GB)...")
    try:
        import gdown
    except ImportError:
        print("  Installing gdown...")
        os.system("pip install gdown")
        import gdown

    kaist_dir = DATA_DIR / "kaist_raw"
    kaist_dir.mkdir(parents=True, exist_ok=True)

    file_id = "1nJkdnSI9fAuhhZPKFfhVXimXGzvuRTb6"
    out_path = kaist_dir / "kaist-cvpr15-preview.tar"

    if out_path.exists():
        print(f"  [SKIP] Already downloaded: {out_path}")
    else:
        gdown.download(id=file_id, output=str(out_path), quiet=False)

    # Extract
    print("  Extracting KAIST archive...")
    extract_dir = kaist_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(out_path, 'r') as tar:
        tar.extractall(extract_dir)
    print(f"  [OK] Extracted to: {extract_dir}")

    # Reorganize into our data/kaist/rgb, data/kaist/thermal structure
    print("  Reorganizing KAIST into rgb/thermal folders...")
    organize_kaist(extract_dir)


def organize_kaist(extract_dir: Path):
    """
    KAIST structure: sets/set00/V000/visible_*.jpg and lwir_*.jpg
    We reorganize to: data/kaist/{split}/rgb/ and data/kaist/{split}/thermal/
    """
    rgb_dir     = DATA_DIR / "kaist" / "train" / "rgb"
    thermal_dir = DATA_DIR / "kaist" / "train" / "thermal"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    thermal_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for vis_path in extract_dir.rglob("*visible*.jpg"):
        therm_path = Path(str(vis_path).replace("visible", "lwir"))
        if therm_path.exists():
            fname = f"{count:06d}.jpg"
            shutil.copy(vis_path, rgb_dir / fname)
            shutil.copy(therm_path, thermal_dir / fname)
            count += 1

    # Also try PNG variants
    for vis_path in extract_dir.rglob("*visible*.png"):
        therm_path = Path(str(vis_path).replace("visible", "lwir"))
        if therm_path.exists():
            fname = f"{count:06d}.png"
            shutil.copy(vis_path, rgb_dir / fname)
            shutil.copy(therm_path, thermal_dir / fname)
            count += 1

    print(f"  [OK] KAIST: organized {count} RGB-Thermal pairs -> {DATA_DIR / 'kaist'}")


# =============================================================================
#  DATASET SUMMARY PRINTER
# =============================================================================

def print_dataset_summary():
    print("\n" + "="*60)
    print("  DATASET SUMMARY")
    print("="*60)
    for ds_name in ["synthetic", "kaist", "cbsr", "neonatal"]:
        ds_path = DATA_DIR / ds_name
        if ds_path.exists():
            for split in ["train", "val", ""]:
                rgb_path = ds_path / split / "rgb" if split else ds_path / "rgb"
                if rgb_path.exists():
                    count = len(list(rgb_path.glob("*.*")))
                    label = f"{ds_name}/{split}" if split else ds_name
                    print(f"  {label:25s}: {count:5d} images")
    print("="*60)


# =============================================================================
#  MAIN
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(description='PC-LiquidGAN Dataset Downloader')
    p.add_argument('--synthetic', action='store_true', help='Generate synthetic dataset only')
    p.add_argument('--kaist',     action='store_true', help='Download KAIST preview only')
    p.add_argument('--all',       action='store_true', help='Download/generate everything')
    p.add_argument('--num_train', type=int, default=500, help='Synthetic train samples')
    p.add_argument('--num_val',   type=int, default=100, help='Synthetic val samples')
    p.add_argument('--img_size',  type=int, default=256, help='Image size for synthetic data')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    DATA_DIR.mkdir(exist_ok=True)

    run_all = args.all or (not args.synthetic and not args.kaist)

    # 1. Always generate synthetic dataset (fast, no internet)
    if args.synthetic or run_all:
        create_synthetic_dataset(
            dataset_name = "synthetic",
            num_train    = args.num_train,
            num_val      = args.num_val,
            img_size     = args.img_size,
        )

    # 2. KAIST (requires internet)
    if args.kaist or run_all:
        try:
            download_kaist_preview()
        except Exception as e:
            print(f"\n[WARN] KAIST download failed: {e}")
            print("  You can manually download from:")
            print("  gdown 1nJkdnSI9fAuhhZPKFfhVXimXGzvuRTb6")
            print("  Then place images in: data/kaist/train/rgb/ and data/kaist/train/thermal/")

    print_dataset_summary()
    print("\n[DONE] To start training with synthetic data:")
    print("       python train.py --dataset synthetic --epochs 50")
    print("\n       To train with KAIST data:")
    print("       python train.py --dataset kaist --epochs 100")
