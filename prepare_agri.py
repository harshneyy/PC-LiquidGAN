"""
prepare_agri.py
Prepares Tomato Leaf Disease dataset for LiquidGAN agriculture domain training.

Input:  RGB plant leaf images (healthy + diseased)
Target: Pseudo-thermal image via green-channel vegetation stress mapping
        (disease areas appear as thermal hotspots, mimicking IR camera output)

Output:
  data/agri/train/rgb/
  data/agri/train/thermal/
  data/agri/val/rgb/
  data/agri/val/thermal/
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

RAW_DIR  = Path("data/agri_raw")
OUT_DIR  = Path("data/agri")
VAL_RATIO = 0.1

def rgb_to_pseudo_thermal(img_rgb: Image.Image) -> Image.Image:
    """
    Convert RGB leaf image to pseudo-thermal heatmap.
    Logic:
      - Plants under stress have reduced photosynthesis (lower green channel)
      - Diseased/stressed regions appear 'hotter' in thermal cameras
      - We invert the green channel (stress = low green = high thermal)
      - Apply a warm colormap to simulate LWIR thermal output
    """
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # Vegetation stress index: higher value = more stressed/diseased
    # Stressed areas: low green, higher red (disease often causes reddish marks)
    stress = np.clip((r * 0.6 - g * 0.5 + 0.3), 0, 1)

    # Map stress to thermal heatmap (cool=blue, warm=red like FLIR cameras)
    thermal_r = np.clip(stress * 1.5, 0, 1)
    thermal_g = np.clip(1.0 - np.abs(stress - 0.5) * 2, 0, 1)
    thermal_b = np.clip((1.0 - stress) * 1.2, 0, 1)

    thermal_arr = np.stack([thermal_r, thermal_g, thermal_b], axis=-1)
    thermal_arr = (thermal_arr * 255).astype(np.uint8)
    return Image.fromarray(thermal_arr)


def prepare():
    # Collect all images from train + valid + test
    all_images = []
    for split_name in ["train", "valid", "test"]:
        img_dir = RAW_DIR / split_name / "images"
        if img_dir.exists():
            all_images.extend(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    all_images = sorted(all_images)
    print(f"Total source images: {len(all_images)}")

    val_cut = int(len(all_images) * (1 - VAL_RATIO))
    splits = {
        "train": all_images[:val_cut],
        "val":   all_images[val_cut:],
    }

    for split, files in splits.items():
        rgb_dir = OUT_DIR / split / "rgb"
        thr_dir = OUT_DIR / split / "thermal"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        thr_dir.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(files):
            fname = f"{i:06d}.jpg"
            try:
                img = Image.open(src).convert("RGB")
                img.save(rgb_dir / fname, quality=95)
                pseudo_thermal = rgb_to_pseudo_thermal(img)
                pseudo_thermal.save(thr_dir / fname, quality=95)
            except Exception as e:
                print(f"  [SKIP] {src.name}: {e}")

        print(f"[OK] {split}: {len(files)} pairs -> {rgb_dir}")

    print("\n=== Agriculture Dataset Summary ===")
    for split in ["train", "val"]:
        count = len(list((OUT_DIR / split / "rgb").glob("*.jpg")))
        print(f"  agri/{split}/rgb: {count} images")

    print("\n[DONE] Run training with:")
    print("  python3 train.py --dataset agri --epochs 100 --batch_size 32")


if __name__ == "__main__":
    prepare()
