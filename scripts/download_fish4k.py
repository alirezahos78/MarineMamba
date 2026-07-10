#!/usr/bin/env python3
"""
Download Fish4Knowledge from GitHub (Callmewuxin/fish4konwledge) and prepare
an 80/20 stratified train/test split in ImageFolder format.

Source: https://github.com/Callmewuxin/fish4konwledge
  23 fish species, 27,370 PNG images. No pre-defined split.

Output layout:
    data/fish4k/
        train/<class_name>/*.png
        test/<class_name>/*.png

Usage:
    cd MarineMamba/
    python3 scripts/download_fish4k.py
    python3 scripts/download_fish4k.py --dst data/fish4k --test-ratio 0.2 --seed 42
"""
import argparse
import io
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from marinemamba.paths import FISH4K_ROOT

GITHUB_ZIP = (
    "https://github.com/Callmewuxin/fish4konwledge/archive/refs/heads/master.zip"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dst", default=FISH4K_ROOT,
                   help="Output root (will contain train/ and test/)")
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--min-test", type=int, default=5,
                   help="Minimum test samples per class (rare classes)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def download_zip(url: str, label: str = "") -> bytes:
    print(f"Downloading {label or url} …")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r:
        total = int(r.headers.get("Content-Length", 0))
        buf = io.BytesIO()
        downloaded = 0
        chunk = 1 << 20  # 1 MB
        while True:
            data = r.read(chunk)
            if not data:
                break
            buf.write(data)
            downloaded += len(data)
            if total:
                pct = downloaded / total * 100
                mb  = downloaded / 1e6
                print(f"\r  {mb:.1f} MB / {total/1e6:.1f} MB  ({pct:.0f}%)", end="", flush=True)
    print()
    return buf.getvalue()


def prepare_split(zip_bytes: bytes, dst: Path, test_ratio: float, min_test: int, seed: int):
    random.seed(seed)
    complete = dst / ".complete"
    if complete.exists():
        print(f"Fish4K already prepared at {dst}")
        return

    print("Extracting ZIP …")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        # Detect repo prefix (fish4konwledge-master/)
        prefix = names[0].split("/")[0] + "/"
        img_entries = [
            n for n in names
            if n.startswith(prefix + "fish_image/")
            and not n.endswith("/")
            and Path(n).suffix.lower() in IMAGE_EXTS
        ]

        # Group by class
        from collections import defaultdict
        class_files = defaultdict(list)
        for n in img_entries:
            parts = n[len(prefix):].split("/")  # fish_image / class / file
            if len(parts) == 3:
                class_files[parts[1]].append(n)

        print(f"Found {len(class_files)} classes, {len(img_entries)} images")

        total_train = total_test = 0
        for cls in sorted(class_files):
            files = class_files[cls]
            random.shuffle(files)
            n_test  = max(min_test, int(len(files) * test_ratio))
            n_test  = min(n_test, len(files) - 1)  # keep at least 1 train image
            n_train = len(files) - n_test
            splits  = {"train": files[:n_train], "test": files[n_train:]}

            for split, split_files in splits.items():
                out_dir = dst / split / cls
                out_dir.mkdir(parents=True, exist_ok=True)
                for zip_name in split_files:
                    fname   = Path(zip_name).name
                    out_path = out_dir / fname
                    if not out_path.exists():
                        data = zf.read(zip_name)
                        out_path.write_bytes(data)

            total_train += len(splits["train"])
            total_test  += len(splits["test"])
            print(f"  {cls:12s}  train={n_train:5d}  test={n_test:4d}")

    complete.write_text(f"fish4k: {total_train} train / {total_test} test\n")
    print(f"\nDone.  {total_train} train / {total_test} test  →  {dst}")


def main():
    args = parse_args()
    dst = Path(args.dst)

    if (dst / ".complete").exists():
        print(f"Fish4K already prepared at {dst}. Skipping download.")
        print("Next step: python3 scripts/build_clip_features.py (with fish4k paths)")
        return

    zip_bytes = download_zip(GITHUB_ZIP, "Fish4Knowledge (GitHub)")
    prepare_split(zip_bytes, dst, args.test_ratio, args.min_test, args.seed)
    print("\nNext step:")
    print("  python3 scripts/build_clip_features.py \\")
    print(f"    --aqua20-root {dst} \\")
    from marinemamba.paths import FISH4K_B16_ROOT, FISH4K_B32_ROOT, FISH4K_CLS_PATH
    print(f"    --out-b16 {FISH4K_B16_ROOT} \\")
    print(f"    --out-b32 {FISH4K_B32_ROOT} \\")
    print(f"    --out-cls {FISH4K_CLS_PATH}")


if __name__ == "__main__":
    main()
