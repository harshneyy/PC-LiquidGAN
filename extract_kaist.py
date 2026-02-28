"""
extract_kaist.py
Efficiently extracts KAIST paired RGB/thermal images from the ZIP.
Reads images directly from ZIP memory (no full disk extraction needed).

Run:
    python extract_kaist.py                    # extract 5000 pairs (fast, ~2 GB)
    python extract_kaist.py --max_pairs 10000  # extract 10000 pairs
    python extract_kaist.py --all              # extract all 50184 pairs (~18 GB)
"""

import os
import zipfile
import argparse
import random
from pathlib import Path

DATA_DIR = Path("./data")
ZIP_PATH = DATA_DIR / "kaist_raw" / "kaist-dataset.zip"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--max_pairs', type=int, default=5000,
                   help='Number of image pairs to extract (default: 5000)')
    p.add_argument('--all', action='store_true',
                   help='Extract all 50184 pairs')
    p.add_argument('--val_ratio', type=float, default=0.1,
                   help='Fraction of pairs for validation (default: 0.1)')
    return p.parse_args()


def extract_kaist(max_pairs=5000, val_ratio=0.1, extract_all=False):
    if not ZIP_PATH.exists():
        print(f"[ERROR] ZIP not found: {ZIP_PATH}")
        return

    print(f"\nKAIST Extractor")
    print(f"{'='*55}")
    print(f"ZIP: {ZIP_PATH}  ({ZIP_PATH.stat().st_size / 1e9:.1f} GB)")

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        all_names = z.namelist()

        # Find all visible images (they have matching lwir counterpart)
        visible_paths = sorted([
            n for n in all_names
            if '/visible/' in n and n.endswith('.jpg')
        ])
        print(f"Total pairs available: {len(visible_paths):,}")

        if not extract_all:
            # Shuffle and pick a balanced subset across all sets
            random.seed(42)
            random.shuffle(visible_paths)
            visible_paths = visible_paths[:max_pairs]

        n_total = len(visible_paths)
        n_val   = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val

        splits = {
            'train': visible_paths[:n_train],
            'val':   visible_paths[n_train:],
        }

        print(f"Extracting: {n_train} train + {n_val} val pairs")
        print(f"Saving to:  data/kaist/train/  and  data/kaist/val/")
        print(f"{'='*55}\n")

        total_extracted = 0
        for split, paths in splits.items():
            rgb_dir = DATA_DIR / "kaist" / split / "rgb"
            thr_dir = DATA_DIR / "kaist" / split / "thermal"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            thr_dir.mkdir(parents=True, exist_ok=True)

            for i, vis_path in enumerate(paths):
                lwir_path = vis_path.replace('/visible/', '/lwir/')
                fname = f"{i:06d}.jpg"

                try:
                    (rgb_dir / fname).write_bytes(z.read(vis_path))
                    (thr_dir / fname).write_bytes(z.read(lwir_path))
                    total_extracted += 1
                except KeyError:
                    continue  # skip missing pair

                if (i + 1) % 500 == 0:
                    pct = (i + 1) / len(paths) * 100
                    print(f"  [{split}] {i+1:5d}/{len(paths):5d}  ({pct:.0f}%)  |  "
                          f"{rgb_dir.name}/  {thr_dir.name}/")

            print(f"  [{split}] DONE  {len(paths):,} pairs")

    print(f"\n{'='*55}")
    print(f"  Extracted {total_extracted:,} RGB-Thermal pairs")
    print(f"  Train: {n_train:,}  |  Val: {n_val:,}")
    print(f"  Saved to: data/kaist/")
    print(f"{'='*55}")
    print(f"\nNext: python train.py --dataset kaist --epochs 100")


if __name__ == '__main__':
    args = get_args()
    extract_kaist(
        max_pairs   = args.max_pairs,
        val_ratio   = args.val_ratio,
        extract_all = args.all,
    )
