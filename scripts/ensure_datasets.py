#!/usr/bin/env python3
"""
Download and prepare the three raw ImageFolder datasets used by SpyMamba.

The script is idempotent: each dataset is skipped when both train/ and test/
class folders already exist.

Examples:
    cd SpyMamba/
    python3 scripts/ensure_datasets.py
    python3 scripts/ensure_datasets.py --datasets fish4k aqua20
    python3 scripts/ensure_datasets.py --sea23-test-ratio 0.2 --seed 42
"""
import argparse
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from spymamba.paths import AQUA20_ROOT, DATA_ROOT, FISH4K_ROOT, SEA23_ROOT

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SEA23_KAGGLE_SLUG = "vencerlanz09/sea-animals-image-dataste"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download missing SpyMamba raw datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("fish4k", "sea23", "aqua20"),
        default=("fish4k", "sea23", "aqua20"),
        help="Datasets to ensure.",
    )
    parser.add_argument("--fish4k-root", type=Path, default=Path(FISH4K_ROOT))
    parser.add_argument("--sea23-root", type=Path, default=Path(SEA23_ROOT))
    parser.add_argument("--aqua20-root", type=Path, default=Path(AQUA20_ROOT))
    parser.add_argument(
        "--sea23-src",
        type=Path,
        default=None,
        help=(
            "Optional already-downloaded Sea Animals raw root containing one "
            "class folder per species. When set, Kaggle download is skipped."
        ),
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path(DATA_ROOT) / "_downloads",
        help="Workspace-local cache for downloaded raw archives/files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fish4k-test-ratio", type=float, default=0.2)
    parser.add_argument("--fish4k-min-test", type=int, default=5)
    parser.add_argument("--sea23-test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--sea23-symlink",
        action="store_true",
        help="Symlink Sea Animals files from the KaggleHub cache instead of copying.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run preparation even if train/test class folders already exist.",
    )
    return parser.parse_args()


def configure_caches(download_dir: Path) -> None:
    cache_root = download_dir / ".cache"
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "huggingface" / "datasets"))
    os.environ.setdefault("KAGGLEHUB_CACHE", str(cache_root / "kagglehub"))
    for key in ("HF_HOME", "HF_DATASETS_CACHE", "KAGGLEHUB_CACHE"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def image_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return [
        item
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS
    ]


def split_has_classes(root: Path, split: str) -> bool:
    split_dir = root / split
    if not split_dir.is_dir():
        return False
    return any(
        child.is_dir() and image_files(child)
        for child in split_dir.iterdir()
    )


def dataset_ready(root: Path) -> bool:
    return split_has_classes(root, "train") and split_has_classes(root, "test")


def download_to_file(url: str, dst: Path, label: str) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"{label} archive exists: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    part_path = dst.with_suffix(dst.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    print(f"Downloading {label} -> {dst}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as response, part_path.open("wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        next_report = 25 * 1024 * 1024
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                if total:
                    pct = downloaded / total * 100.0
                    print(f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", flush=True)
                else:
                    print(f"  {downloaded / 1e6:.1f} MB", flush=True)
                next_report += 25 * 1024 * 1024

    part_path.replace(dst)
    print(f"Downloaded {label}: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def ensure_fish4k(args: argparse.Namespace) -> None:
    if dataset_ready(args.fish4k_root) and not args.force:
        print(f"Fish4Knowledge exists: {args.fish4k_root}")
        return

    from scripts.download_fish4k import GITHUB_ZIP, prepare_split

    print(f"Preparing Fish4Knowledge at {args.fish4k_root}")
    zip_path = download_to_file(
        GITHUB_ZIP,
        args.download_dir / "fish4knowledge_master.zip",
        "Fish4Knowledge (GitHub)",
    )
    prepare_split(
        zip_path.read_bytes(),
        args.fish4k_root,
        args.fish4k_test_ratio,
        args.fish4k_min_test,
        args.seed,
    )


def ensure_aqua20(args: argparse.Namespace) -> None:
    if dataset_ready(args.aqua20_root) and not args.force:
        print(f"AQUA20 exists: {args.aqua20_root}")
        return

    try:
        from scripts.download_aqua20 import download_aqua20
    except ImportError as exc:
        raise RuntimeError(
            "AQUA20 download requires the 'datasets' package. "
            "Install it with: python3 -m pip install datasets"
        ) from exc

    print(f"Preparing AQUA20 at {args.aqua20_root}")
    download_aqua20(str(args.aqua20_root))


def unpack_nested_zips(root: Path, extract_root: Path) -> List[Path]:
    roots = [root]
    for zip_path in sorted(root.rglob("*.zip")):
        out_dir = extract_root / zip_path.stem
        marker = out_dir / ".extracted"
        if not marker.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(out_dir)
            marker.write_text(f"extracted from {zip_path}\n")
        roots.append(out_dir)
    return roots


def class_root_score(path: Path) -> Tuple[int, int]:
    if not path.is_dir():
        return (0, 0)
    class_count = 0
    image_count = 0
    for child in path.iterdir():
        if not child.is_dir():
            continue
        n_images = len(image_files(child))
        if n_images:
            class_count += 1
            image_count += n_images
    return class_count, image_count


def find_class_root(candidates: Sequence[Path]) -> Path:
    best_path: Optional[Path] = None
    best_score = (0, 0)
    for root in candidates:
        search_roots = [root] + [p for p in root.rglob("*") if p.is_dir()]
        for path in search_roots:
            score = class_root_score(path)
            if score > best_score:
                best_path = path
                best_score = score

    if best_path is None or best_score[0] == 0:
        searched = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"Could not find Sea Animals class folders under: {searched}")
    return best_path


def prepare_sea23_split(
    src: Path,
    dst: Path,
    test_ratio: float,
    seed: int,
    symlink: bool,
) -> None:
    random.seed(seed)
    class_dirs = sorted([d for d in src.iterdir() if d.is_dir() and image_files(d)])
    if not class_dirs:
        raise RuntimeError(f"No image class subdirectories found in {src}")

    total_train = 0
    total_test = 0
    print(f"Sea Animals source: {src}")
    print(f"Sea Animals output: {dst}")
    print(f"Classes: {len(class_dirs)}")

    for cls_dir in class_dirs:
        images = image_files(cls_dir)
        random.shuffle(images)
        n_test = max(1, int(len(images) * test_ratio))
        n_train = len(images) - n_test
        splits = {"train": images[:n_train], "test": images[n_train:]}

        for split, files in splits.items():
            out_dir = dst / split / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for src_file in files:
                dst_file = out_dir / src_file.name
                if dst_file.exists() or dst_file.is_symlink():
                    dst_file.unlink()
                if symlink:
                    dst_file.symlink_to(src_file.resolve())
                else:
                    shutil.copy2(src_file, dst_file)

        total_train += n_train
        total_test += n_test
        print(f"  {cls_dir.name:30s}  train={n_train:4d}  test={n_test:4d}")

    (dst / ".complete").write_text(
        f"sea23: {total_train} train / {total_test} test\n"
        f"source: {src}\n"
    )
    print(f"Done. Sea Animals: {total_train} train / {total_test} test")


def ensure_sea23(args: argparse.Namespace) -> None:
    if dataset_ready(args.sea23_root) and not args.force:
        print(f"Sea Animals exists: {args.sea23_root}")
        return

    if args.sea23_src is not None:
        src = find_class_root([args.sea23_src])
    else:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError(
                "Sea Animals download requires the 'kagglehub' package. "
                "Install it with: python3 -m pip install kagglehub"
            ) from exc

        print(f"Downloading Sea Animals 23 from Kaggle: {SEA23_KAGGLE_SLUG}")
        try:
            downloaded = Path(kagglehub.dataset_download(SEA23_KAGGLE_SLUG))
        except Exception as exc:
            raise RuntimeError(
                "Kaggle denied access to Sea Animals 23. Authenticate Kaggle "
                "first (for example with ~/.kaggle/kaggle.json) and accept the "
                f"dataset terms at https://www.kaggle.com/datasets/{SEA23_KAGGLE_SLUG}, "
                "or pass an already-downloaded raw directory with --sea23-src."
            ) from exc
        extract_root = args.download_dir / "sea23_extracted"
        candidate_roots = unpack_nested_zips(downloaded, extract_root)
        src = find_class_root(candidate_roots)

    prepare_sea23_split(
        src,
        args.sea23_root,
        args.sea23_test_ratio,
        args.seed,
        args.sea23_symlink,
    )


def main() -> None:
    args = parse_args()
    args.download_dir.mkdir(parents=True, exist_ok=True)
    configure_caches(args.download_dir)

    handlers = {
        "fish4k": ensure_fish4k,
        "sea23": ensure_sea23,
        "aqua20": ensure_aqua20,
    }
    for name in args.datasets:
        handlers[name](args)

    print("Dataset check complete.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
