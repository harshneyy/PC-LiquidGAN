"""
kaggle_download.py
Downloads KAIST dataset from Kaggle using the Kaggle API v2.

Dataset: https://www.kaggle.com/datasets/adlteam/kaist-dataset

Usage:
    python kaggle_download.py              # Download dataset zip + organize
    python kaggle_download.py --list       # Just list available files
    python kaggle_download.py --organize   # Only organize (if already downloaded)
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path

DATA_DIR    = Path("./data")
RAW_DIR     = DATA_DIR / "kaist_raw"
DATASET_SLUG = "adlteam/kaist-dataset"


# ─────────────────────────────────────────────────────────────────────────────

def get_api():
    """Authenticate and return Kaggle API object (v2)."""
    try:
        from kaggle import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("[OK] Kaggle authentication successful!")
        return api
    except Exception as e:
        print(f"[ERROR] Kaggle auth failed: {e}")
        print("\nMake sure kaggle.json is at: C:\\Users\\SIC-LAB\\.kaggle\\kaggle.json")
        sys.exit(1)


def list_files(api):
    """List all files in the KAIST dataset on Kaggle."""
    print(f"\nListing files in: {DATASET_SLUG}")
    files = api.dataset_list_files(DATASET_SLUG).files
    print(f"Total files: {len(files)}")
    for f in files:
        print(f"  {str(f.name)}")
    return files


def download_dataset(api):
    """Download the KAIST dataset zip from Kaggle."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[Kaggle Download] Downloading KAIST dataset...")
    print(f"  Destination: {RAW_DIR.resolve()}")
    print(f"  NOTE: This may take a while depending on your internet speed.")
    print(f"  (The full dataset is ~20 GB, downloading as ZIP)\n")

    api.dataset_download_files(
        DATASET_SLUG,
        path=str(RAW_DIR),
        unzip=False,   # We'll unzip manually for control
        quiet=False,   # Show progress bar
        force=False,   # Skip if already downloaded
    )

    # Find the downloaded zip
    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        print("[WARN] No ZIP file found after download. Check the RAW_DIR for contents.")
        return None
    print(f"\n[OK] Downloaded: {zips[0].name}  ({zips[0].stat().st_size / 1e9:.2f} GB)")
    return zips[0]


def extract_and_organize(zip_path: Path, max_pairs: int = 2000):
    """
    Extract paired visible (RGB) and lwir (thermal) images.
    
    KAIST structure inside ZIP:
        set00/V000/visible/I00000.jpg   <- RGB image
        set00/V000/lwir/I00000.jpg      <- Thermal image

    Output structure:
        data/kaist/train/rgb/           <- visible images
        data/kaist/train/thermal/       <- lwir images
        data/kaist/val/rgb/             
        data/kaist/val/thermal/         
    """
    print(f"\n[Extracting] {zip_path.name} ...")

    extract_dir = RAW_DIR / "extracted"
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        all_members = z.namelist()
        # Filter only visible + lwir images (skip annotations etc.)
        visible_files = [m for m in all_members if '/visible/' in m and m.endswith('.jpg')]
        lwir_files    = {m.replace('/visible/', '/lwir/'): m for m in visible_files}

        # Pair them up
        pairs = [(v, v.replace('/visible/', '/lwir/')) 
                 for v in visible_files 
                 if v.replace('/visible/', '/lwir/') in all_members]

        print(f"  Found {len(pairs)} RGB-Thermal pairs in ZIP")
        print(f"  Extracting up to {max_pairs} pairs (to save disk space)...")

        # Organize into train (90%) and val (10%)
        train_cut = int(min(len(pairs), max_pairs) * 0.9)
        splits = {
            'train': pairs[:train_cut],
            'val':   pairs[train_cut:min(len(pairs), max_pairs)],
        }

        for split, split_pairs in splits.items():
            rgb_dir = DATA_DIR / "kaist" / split / "rgb"
            thr_dir = DATA_DIR / "kaist" / split / "thermal"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            thr_dir.mkdir(parents=True, exist_ok=True)

            for i, (vis_path, lwir_path) in enumerate(split_pairs):
                fname = f"{i:06d}.jpg"
                # Extract directly into memory and save
                try:
                    with z.open(vis_path)  as src: (rgb_dir / fname).write_bytes(src.read())
                    with z.open(lwir_path) as src: (thr_dir / fname).write_bytes(src.read())
                except KeyError:
                    continue

                if (i + 1) % 200 == 0:
                    print(f"    [{split}] {i+1}/{len(split_pairs)} done")

            print(f"  [OK] {split}: {len(split_pairs)} pairs -> {rgb_dir}")

    return True


# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="KAIST Kaggle Downloader")
    p.add_argument('--list',      action='store_true', help='Only list dataset files')
    p.add_argument('--organize',  action='store_true', help='Only organize (already downloaded)')
    p.add_argument('--max_pairs', type=int, default=2000,
                   help='Max image pairs to extract (default: 2000, ~200 MB)')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()

    print("=" * 60)
    print("  KAIST Dataset Kaggle Downloader")
    print("=" * 60)

    api = get_api()

    if args.list:
        list_files(api)
        sys.exit(0)

    if args.organize:
        zips = list(RAW_DIR.glob("*.zip"))
        if not zips:
            print("[ERROR] No ZIP found in data/kaist_raw/. Run without --organize first.")
            sys.exit(1)
        extract_and_organize(zips[0], max_pairs=args.max_pairs)
    else:
        zip_path = download_dataset(api)
        if zip_path:
            extract_and_organize(zip_path, max_pairs=args.max_pairs)

    # Print summary
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    for split in ["train", "val"]:
        rgb_path = DATA_DIR / "kaist" / split / "rgb"
        if rgb_path.exists():
            count = len(list(rgb_path.glob("*.jpg")))
            print(f"  kaist/{split:5s}/rgb     : {count:5d} images")
    print("=" * 60)
    print("\n[DONE] Start training:")
    print("       python train.py --dataset kaist --epochs 100")
