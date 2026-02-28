"""
create_domain_splits.py
Creates CBSR and Neonatal placeholder datasets from KAIST sub-sets.

KAIST has 12 sets (set00-set11), each captured in different conditions.
We use different sets to simulate different thermal domains:
  - data/kaist/   -> set00-set07 (daytime outdoor pedestrian)
  - data/cbsr/    -> set08-set09 (nighttime, different distribution)
  - data/neonatal/-> set10-set11 (different scene type)

This allows zero-shot cross-domain evaluation on unseen domains.

Run:
    python create_domain_splits.py
"""

import zipfile
import random
from pathlib import Path

DATA_DIR = Path("./data")
ZIP_PATH = DATA_DIR / "kaist_raw" / "kaist-dataset.zip"

DOMAIN_SETS = {
    "cbsr":     ["set08", "set09"],          # nighttime scenes
    "neonatal": ["set10", "set11"],          # different conditions
}
PAIRS_PER_DOMAIN = 500   # 450 train + 50 val each


def extract_domain(z, set_names, domain_name, n_pairs=500, val_ratio=0.1):
    """Extract images from specific KAIST sets into a named domain folder."""
    all_visible = sorted([
        n for n in z.namelist()
        if '/visible/' in n and n.endswith('.jpg')
        and any(s in n for s in set_names)
    ])

    if not all_visible:
        print(f"  [WARN] No images found for sets: {set_names}")
        return 0

    random.seed(42)
    random.shuffle(all_visible)
    selected = all_visible[:n_pairs]

    n_val   = max(1, int(len(selected) * val_ratio))
    n_train = len(selected) - n_val
    splits  = {'train': selected[:n_train], 'val': selected[n_train:]}

    total = 0
    for split, paths in splits.items():
        rgb_dir = DATA_DIR / domain_name / split / "rgb"
        thr_dir = DATA_DIR / domain_name / split / "thermal"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        thr_dir.mkdir(parents=True, exist_ok=True)

        for i, vis_path in enumerate(paths):
            lwir_path = vis_path.replace('/visible/', '/lwir/')
            fname = f"{i:05d}.jpg"
            try:
                (rgb_dir / fname).write_bytes(z.read(vis_path))
                (thr_dir / fname).write_bytes(z.read(lwir_path))
                total += 1
            except KeyError:
                continue

        print(f"  [{domain_name}/{split}] {len(paths)} pairs saved")

    return total


def main():
    print("Creating domain-split datasets from KAIST...")
    print("="*55)

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for domain, sets in DOMAIN_SETS.items():
            # Check if already exists
            existing = DATA_DIR / domain / "train" / "rgb"
            if existing.exists() and len(list(existing.glob("*.jpg"))) > 0:
                count = len(list(existing.glob("*.jpg")))
                print(f"[SKIP] {domain}: already has {count} images")
                continue

            print(f"\nCreating '{domain}' domain from sets {sets}:")
            count = extract_domain(z, sets, domain, n_pairs=PAIRS_PER_DOMAIN)
            print(f"  Total extracted: {count} pairs")

    print("\n" + "="*55)
    print("Domain datasets ready:")
    for domain in list(DOMAIN_SETS.keys()) + ["kaist", "synthetic"]:
        for split in ["train", "val"]:
            p = DATA_DIR / domain / split / "rgb"
            if p.exists():
                n = len(list(p.glob("*.*")))
                if n > 0:
                    print(f"  data/{domain}/{split}/rgb/  : {n:5d} images")
    print("="*55)
    print("\nTraining command:")
    print("  python train.py --dataset kaist --epochs 100")
    print("\nZero-shot evaluation (after training):")
    print("  python evaluate.py --checkpoint checkpoints/ckpt_epoch_0100.pth --dataset cbsr --unpaired")
    print("  python evaluate.py --checkpoint checkpoints/ckpt_epoch_0100.pth --dataset neonatal --unpaired")


if __name__ == '__main__':
    main()
