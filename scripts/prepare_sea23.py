#!/usr/bin/env python3
"""
prepare_sea23.py — Prepare the Sea Animals 23 dataset (vencerlanz09/sea-animals-image-dataste).

The raw Kaggle download is a flat directory of 23 class folders (no train/test split).
This script performs an 80/20 stratified split and writes:

    data/sea23/
        train/<class>/<image>
        test/<class>/<image>

Usage:
    cd MarineMamba/
    python3 scripts/prepare_sea23.py --src /path/to/raw/sea-animals

    # With a custom output dir:
    python3 scripts/prepare_sea23.py --src /path/to/raw/sea-animals --dst data/sea23

    # Set a reproducible seed:
    python3 scripts/prepare_sea23.py --src /path/to/raw/sea-animals --seed 42
"""
import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--src", required=True,
                   help="Root of the raw sea-animals download (contains one folder per class).")
    p.add_argument("--dst", default=str(Path(__file__).resolve().parents[1] / "data" / "sea23"),
                   help="Output root for train/test split.")
    p.add_argument("--test-ratio", type=float, default=0.2,
                   help="Fraction of images to hold out for the test set.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--symlink", action="store_true",
                   help="Symlink images instead of copying (saves disk space).")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)

    class_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class subdirectories found in {src}")

    print(f"Source : {src}")
    print(f"Output : {dst}")
    print(f"Classes: {len(class_dirs)}")
    print(f"Split  : {int((1 - args.test_ratio) * 100)}/{int(args.test_ratio * 100)} train/test\n")

    total_train = total_test = 0

    for cls_dir in class_dirs:
        images = sorted([f for f in cls_dir.iterdir()
                         if f.is_file() and f.suffix.lower() in IMAGE_EXTS])
        if not images:
            print(f"  [warn] No images in {cls_dir.name}, skipping.")
            continue

        random.shuffle(images)
        n_test  = max(1, int(len(images) * args.test_ratio))
        n_train = len(images) - n_test
        splits  = {"train": images[:n_train], "test": images[n_train:]}

        for split, files in splits.items():
            out_dir = dst / split / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                dst_file = out_dir / f.name
                if dst_file.exists() or dst_file.is_symlink():
                    dst_file.unlink()
                if args.symlink:
                    dst_file.symlink_to(f.resolve())
                else:
                    shutil.copy2(f, dst_file)

        total_train += len(splits["train"])
        total_test  += len(splits["test"])
        print(f"  {cls_dir.name:30s}  train={len(splits['train']):4d}  test={len(splits['test']):4d}")

    print(f"\nDone.  Total: {total_train} train / {total_test} test  →  {dst}")


if __name__ == "__main__":
    main()
