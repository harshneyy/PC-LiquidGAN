"""
prepare_medical.py
Prepares Knee Osteoarthritis (X-Ray + Thermal) dataset for LiquidGAN training.

Dataset structure:
  data/medical_raw/Dataset/X-Ray  Images/{0,1,2,3,4}/  <- RGB input
  data/medical_raw/Dataset/Thermal Images/{0,1,2,3,4}/  <- Thermal target

Filenames are matched by patient ID (same stem without 'heatmap_' prefix).

Output:
  data/medical/train/rgb/
  data/medical/train/thermal/
  data/medical/val/rgb/
  data/medical/val/thermal/
"""

import os
from pathlib import Path
from PIL import Image

RAW_DIR  = Path("data/medical_raw/Dataset")
OUT_DIR  = Path("data/medical")
GRADES   = ["0", "1", "2", "3", "4"]
VAL_RATIO = 0.1
MAX_PAIRS = 5000  # cap to save training time

def get_pairs():
    """Match X-Ray and Thermal images by patient ID across all grades."""
    pairs = []
    for grade in GRADES:
        xray_dir    = RAW_DIR / "X-Ray  Images" / grade
        thermal_dir = RAW_DIR / "Thermal Images" / grade

        xray_files = {f.stem: f for f in xray_dir.glob("*.png")}
        # Thermal files are prefixed with 'heatmap_'
        for t_file in thermal_dir.glob("*.png"):
            patient_id = t_file.stem.replace("heatmap_", "")
            if patient_id in xray_files:
                pairs.append((xray_files[patient_id], t_file))

    print(f"Total matched pairs: {len(pairs)}")
    return pairs


def prepare():
    pairs = get_pairs()
    pairs = pairs[:MAX_PAIRS]

    val_cut = int(len(pairs) * (1 - VAL_RATIO))
    splits = {
        "train": pairs[:val_cut],
        "val":   pairs[val_cut:],
    }

    for split, split_pairs in splits.items():
        rgb_dir = OUT_DIR / split / "rgb"
        thr_dir = OUT_DIR / split / "thermal"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        thr_dir.mkdir(parents=True, exist_ok=True)

        for i, (xray_path, thermal_path) in enumerate(split_pairs):
            fname = f"{i:06d}.jpg"
            try:
                # X-Ray → convert to RGB
                Image.open(xray_path).convert("RGB").save(rgb_dir / fname, quality=95)
                # Thermal heatmap → keep as RGB (already colorized thermal)
                Image.open(thermal_path).convert("RGB").save(thr_dir / fname, quality=95)
            except Exception as e:
                print(f"  [SKIP] {xray_path.name}: {e}")
                continue

            if (i + 1) % 500 == 0:
                print(f"  [{split}] {i+1}/{len(split_pairs)} done")

        print(f"[OK] {split}: {len(split_pairs)} pairs -> {rgb_dir}")

    print("\n=== Medical Dataset Summary ===")
    for split in ["train", "val"]:
        count = len(list((OUT_DIR / split / "rgb").glob("*.jpg")))
        print(f"  medical/{split}/rgb: {count} images")

    print("\n[DONE] Run training with:")
    print("  python3 train.py --dataset medical --epochs 100 --batch_size 32")


if __name__ == "__main__":
    prepare()
