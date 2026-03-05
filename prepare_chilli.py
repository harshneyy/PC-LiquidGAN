"""
prepare_chilli.py
Prepares Chilli/Pepper Leaf Disease dataset for LiquidGAN training.

Same thermal synthesis approach as agri (vegetation stress index).
Input: RGB pepper leaf images (Bacterial Spot + Healthy)
Target: Pseudo-thermal via green-channel stress mapping

Output:
  data/chilli/train/rgb/
  data/chilli/train/thermal/
  data/chilli/val/rgb/
  data/chilli/val/thermal/
"""

import numpy as np
from pathlib import Path
from PIL import Image

RAW_DIR   = Path("data/chilli_raw")
OUT_DIR   = Path("data/chilli")
VAL_RATIO = 0.1
CLASSES   = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy"]

def rgb_to_pseudo_thermal(img_rgb: Image.Image) -> Image.Image:
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    # Diseased leaves: reduced green, more thermal emission
    stress = np.clip((r * 0.6 - g * 0.5 + 0.3), 0, 1)
    thermal_r = np.clip(stress * 1.5, 0, 1)
    thermal_g = np.clip(1.0 - np.abs(stress - 0.5) * 2, 0, 1)
    thermal_b = np.clip((1.0 - stress) * 1.2, 0, 1)
    thermal_arr = np.stack([thermal_r, thermal_g, thermal_b], axis=-1)
    return Image.fromarray((thermal_arr * 255).astype(np.uint8))


def prepare():
    # Collect all images from train + val + test splits
    all_images = []
    for split_name in ["train", "val", "test"]:
        for cls in CLASSES:
            cls_dir = RAW_DIR / split_name / cls
            if cls_dir.exists():
                all_images.extend(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.JPG")) + list(cls_dir.glob("*.png")))

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
                rgb_to_pseudo_thermal(img).save(thr_dir / fname, quality=95)
            except Exception as e:
                print(f"  [SKIP] {src.name}: {e}")

        print(f"[OK] {split}: {len(files)} pairs -> {rgb_dir}")

    print("\n=== Chilli Dataset Summary ===")
    for split in ["train", "val"]:
        count = len(list((OUT_DIR / split / "rgb").glob("*.jpg")))
        print(f"  chilli/{split}/rgb: {count} images")

    print("\n[DONE] Run training with:")
    print("  python3 train.py --dataset chilli --epochs 100 --batch_size 32")


if __name__ == "__main__":
    prepare()
