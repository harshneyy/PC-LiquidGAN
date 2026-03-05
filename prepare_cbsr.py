"""
prepare_cbsr.py
Prepares CBSR NIR Face dataset for LiquidGAN training.

Strategy:
  - Input  (RGB channel): NIR probe images  (full near-infrared)
  - Target (thermal):     Grayscale of same image (simulates thermal intensity map)
  - This is valid for cross-domain evaluation as per the paper.

Output structure:
  data/cbsr/train/rgb/       <- NIR probe images (as RGB via 3-channel repeat)
  data/cbsr/train/thermal/   <- Grayscale of same (pseudo-thermal)
  data/cbsr/val/rgb/
  data/cbsr/val/thermal/
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

SRC_DIR   = Path("data/cbsr_raw/NIR_face_dataset/NIR_face_dataset")
OUT_DIR   = Path("data/cbsr")
VAL_RATIO = 0.1
MAX_PAIRS = 3000  # enough for training

def prepare():
    # Get all probe (non-gallery) images
    all_files = sorted(SRC_DIR.glob("*.bmp"))
    probe_files = [f for f in all_files if not f.stem.endswith("-g")]

    print(f"Found {len(probe_files)} probe (NIR) images")
    probe_files = probe_files[:MAX_PAIRS]

    val_cut = int(len(probe_files) * (1 - VAL_RATIO))
    splits = {
        "train": probe_files[:val_cut],
        "val":   probe_files[val_cut:],
    }

    for split, files in splits.items():
        rgb_dir = OUT_DIR / split / "rgb"
        thr_dir = OUT_DIR / split / "thermal"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        thr_dir.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(files):
            fname = f"{i:06d}.jpg"
            img = Image.open(src).convert("L")  # grayscale NIR

            # RGB: stack grayscale 3x to create pseudo-RGB (NIR intensity)
            rgb_img = Image.merge("RGB", [img, img, img])
            rgb_img.save(rgb_dir / fname, quality=95)

            # Thermal: apply a mild temperature-like colormap via numpy
            arr = np.array(img, dtype=np.float32) / 255.0
            # Slightly boost mid-range to simulate thermal distribution
            thermal_arr = np.clip(arr * 1.1 + 0.05 * (arr ** 2), 0, 1)
            thermal_img = Image.fromarray((thermal_arr * 255).astype(np.uint8)).convert("L")
            thermal_rgb = Image.merge("RGB", [thermal_img, thermal_img, thermal_img])
            thermal_rgb.save(thr_dir / fname, quality=95)

            if (i + 1) % 300 == 0:
                print(f"  [{split}] {i+1}/{len(files)} done")

        print(f"[OK] {split}: {len(files)} pairs -> {rgb_dir}")

    # Summary
    print("\n=== CBSR Dataset Summary ===")
    for split in ["train", "val"]:
        count = len(list((OUT_DIR / split / "rgb").glob("*.jpg")))
        print(f"  cbsr/{split}/rgb: {count} images")

if __name__ == "__main__":
    prepare()
    print("\n[DONE] Run training with:")
    print("  python3 train.py --dataset cbsr --epochs 100 --batch_size 32")
