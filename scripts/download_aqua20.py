#!/usr/bin/env python3
"""
Download AQUA20 from Hugging Face and save it as an ImageFolder dataset.

Output layout:
    spymamba/data/aqua20/
        train/<class_name>/*.jpg
        test/<class_name>/*.jpg

Usage:
    python3 scripts/download_aqua20.py
    python3 scripts/download_aqua20.py --output-dir /custom/path/aqua20
"""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from spymamba.paths import AQUA20_ROOT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=AQUA20_ROOT,
                   help=f"Where to save AQUA20 images (default: {AQUA20_ROOT})")
    return p.parse_args()


def download_aqua20(output_dir):
    train_dir     = os.path.join(output_dir, "train")
    test_dir      = os.path.join(output_dir, "test")
    complete_flag = os.path.join(output_dir, ".complete")

    if os.path.isfile(complete_flag):
        print(f"AQUA20 already downloaded at {output_dir}")
        return train_dir, test_dir

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Install it with:")
        print("    pip install datasets")
        sys.exit(1)

    print("Downloading AQUA20 from Hugging Face (taufiktrf/AQUA20)...")
    dataset     = load_dataset("taufiktrf/AQUA20")
    class_names = dataset["train"].features["label"].names
    print(f"Classes ({len(class_names)}): {class_names}")

    for split_name, split_dir in [("train", train_dir), ("test", test_dir)]:
        for cls in class_names:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

        print(f"Saving {split_name} split ...")
        for i, example in enumerate(dataset[split_name]):
            label     = int(example["label"])
            img_path  = os.path.join(split_dir, class_names[label], f"{i:06d}.jpg")
            if not os.path.isfile(img_path):
                example["image"].convert("RGB").save(img_path, quality=95)
        n = len(dataset[split_name])
        print(f"  {split_name}: {n} images saved")

    with open(complete_flag, "w") as f:
        f.write(f"AQUA20: {len(dataset['train'])} train, {len(dataset['test'])} test\n")

    print(f"\nDone. AQUA20 saved to {output_dir}")
    print("Next step: python3 scripts/build_clip_features.py")
    return train_dir, test_dir


def main():
    args = parse_args()
    download_aqua20(args.output_dir)


if __name__ == "__main__":
    main()
