#!/usr/bin/env python3
"""
Create a clustered stacked bar chart for per-class train/test sample counts.

Default chart:
  - Datasets: Fish4Knowledge, Sea Animals, AQUA20
  - X axis: class slots; each slot contains 3 bars, one per dataset
  - Y axis: total sample count for that class
  - Bar body: train/test percentages, stacked inside each dataset bar

Examples:
    cd SpyMamba/

    # Count raw ImageFolder samples from data/{fish4k,sea23,aqua20}
    python3 scripts/clustered_class_barchart.py

    # Force 23 class slots -> 23 * 3 = 69 bar positions
    python3 scripts/clustered_class_barchart.py --max-class-slots 23

    # Show class labels if needed; labels are hidden by default
    python3 scripts/clustered_class_barchart.py --max-class-slots 23 --class-labels

    # Make very small classes easier to see under heavy imbalance
    python3 scripts/clustered_class_barchart.py --max-class-slots 23 --yscale log1p

    # Override raw dataset roots
    python3 scripts/clustered_class_barchart.py \
        --dataset "Fish4Knowledge=/path/to/fish4k" \
        --dataset "Sea Animals=/path/to/sea23" \
        --dataset "AQUA20=/path/to/aqua20"

    # Count from feature caches instead of raw image folders
    python3 scripts/clustered_class_barchart.py --source features

    # Count from a CSV with columns: dataset,class,train,test[,class_index]
    python3 scripts/clustered_class_barchart.py --source csv --csv counts.csv
"""
import argparse
import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from spymamba.config import get_config
from spymamba.paths import AQUA20_ROOT, FISH4K_ROOT, LOGS_DIR, SEA23_ROOT

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Optional[Path] = None
    config: Optional[str] = None


@dataclass
class ClassCount:
    dataset: str
    class_index: int
    class_name: str
    train: float
    test: float

    @property
    def total(self) -> float:
        return self.train + self.test

    @property
    def train_pct(self) -> float:
        return 100.0 * self.train / self.total if self.total > 0 else 0.0

    @property
    def test_pct(self) -> float:
        return 100.0 * self.test / self.total if self.total > 0 else 0.0


DEFAULT_DATASETS = [
    DatasetSpec("Fish4Knowledge", Path(FISH4K_ROOT), "fish4k_baseline"),
    DatasetSpec("Sea Animals", Path(SEA23_ROOT), "sea23_pyramid_hybrid_128_focal_balanced"),
    DatasetSpec("AQUA20", Path(AQUA20_ROOT), "aqua20_pyramid_hybrid_128_focal_balanced"),
]

DEFAULT_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#17becf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Clustered stacked class-count bar chart for aquatic datasets.",
    )
    parser.add_argument(
        "--source",
        choices=("raw", "features", "csv"),
        default="raw",
        help="Where to read class counts from.",
    )
    parser.add_argument(
        "--download-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For the built-in raw datasets, download/prepare missing datasets "
            "before plotting."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="NAME=ROOT",
        help=(
            "Raw ImageFolder dataset root. Repeat for each dataset. "
            "When provided, replaces the built-in three dataset roots."
        ),
    )
    parser.add_argument(
        "--feature-config",
        action="append",
        default=None,
        metavar="NAME=CONFIG",
        help=(
            "Named spymamba config to count from feature caches. Repeat for each dataset. "
            "When provided, replaces the built-in three configs."
        ),
    )
    parser.add_argument(
        "--feature-count-mode",
        choices=("unique-paths", "labels"),
        default="unique-paths",
        help=(
            "For feature caches, count unique image paths when available, or count labels. "
            "unique-paths avoids counting augmented train views as separate samples."
        ),
    )
    parser.add_argument(
        "--feature-train-view-factor",
        type=float,
        default=4.0,
        help="Divide train label counts by this factor when --feature-count-mode labels is used.",
    )
    parser.add_argument(
        "--feature-test-view-factor",
        type=float,
        default=1.0,
        help="Divide test label counts by this factor when --feature-count-mode labels is used.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV input for --source csv. Required columns: dataset,class,train,test.",
    )
    parser.add_argument(
        "--max-class-slots",
        type=int,
        default=None,
        help=(
            "Number of class slots on the x axis. Default is the maximum class count "
            "among datasets. Use 23 for exactly 69 positions with 3 datasets."
        ),
    )
    parser.add_argument(
        "--sort-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Order 3-bar class batches from high to low by the selected count metric."
        ),
    )
    parser.add_argument(
        "--sort-batch-metric",
        choices=("class-count", "aggregate", "max"),
        default="class-count",
        help=(
            "Metric used by --sort-batches: class-count/aggregate sums train+test "
            "counts across the 3 dataset bars in a batch; max uses the tallest "
            "dataset bar in that batch."
        ),
    )
    parser.add_argument(
        "--class-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show x-axis class slot labels. Disabled by default to keep 3-bar batches clean.",
    )
    parser.add_argument(
        "--x-label-mode",
        choices=("index", "names"),
        default="index",
        help="Class label style when --class-labels is enabled.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(LOGS_DIR) / "clustered_class_distribution.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional output CSV for the counts used by the figure.",
    )
    parser.add_argument(
        "--no-summary-csv",
        action="store_true",
        help="Do not write the default summary CSV next to the figure.",
    )
    parser.add_argument("--title", default="Per-class sample distribution by dataset")
    parser.add_argument("--fig-width", type=float, default=14.0)
    parser.add_argument("--fig-height", type=float, default=7.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--bar-width", type=float, default=0.24)
    parser.add_argument(
        "--x-margin",
        type=float,
        default=0.35,
        help="Horizontal margin around the first/last class slot; smaller values zoom in.",
    )
    parser.add_argument(
        "--y-padding",
        type=float,
        default=0.04,
        help="Fraction of extra headroom above the tallest bar; smaller values zoom in.",
    )
    parser.add_argument(
        "--yscale",
        choices=("linear", "sqrt", "log", "log1p"),
        default="sqrt",
        help=(
            "Height transform for sample counts. sqrt is the default so rare "
            "classes remain visible; log/log1p use a safe log(1+x) transform."
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=("auto", "all", "none"),
        default="auto",
        help="Percentage labels inside train/test bar segments.",
    )
    parser.add_argument(
        "--label-min-fraction",
        type=float,
        default=0.025,
        help="In auto mode, only label segments at least this fraction of the y max.",
    )
    parser.add_argument("--percent-decimals", type=int, default=0)
    parser.add_argument(
        "--show-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark missing dataset/class slots as n/a at the baseline.",
    )
    return parser.parse_args()


def parse_name_value(items: Optional[Sequence[str]], value_label: str) -> Optional[List[Tuple[str, str]]]:
    if not items:
        return None
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME={value_label}, got: {item}")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Expected NAME={value_label}, got: {item}")
        parsed.append((name, value))
    return parsed


def dataset_specs_from_args(args: argparse.Namespace) -> List[DatasetSpec]:
    if args.source == "raw":
        overrides = parse_name_value(args.dataset, "ROOT")
        if overrides:
            return [DatasetSpec(name, Path(root), None) for name, root in overrides]
        return DEFAULT_DATASETS

    if args.source == "features":
        overrides = parse_name_value(args.feature_config, "CONFIG")
        if overrides:
            return [DatasetSpec(name, None, config_name) for name, config_name in overrides]
        return DEFAULT_DATASETS

    return []


def count_images_in_class_dir(class_dir: Path) -> int:
    return sum(
        1
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def class_dirs(split_dir: Path) -> Dict[str, Path]:
    if not split_dir.exists():
        return {}
    return {
        item.name: item
        for item in sorted(split_dir.iterdir())
        if item.is_dir()
    }


def load_raw_counts(spec: DatasetSpec) -> List[ClassCount]:
    if spec.root is None:
        raise ValueError(f"No raw root configured for {spec.name}")
    root = spec.root
    train_dirs = class_dirs(root / "train")
    test_dirs = class_dirs(root / "test")
    if not train_dirs and not test_dirs:
        raise FileNotFoundError(
            f"No train/ or test/ class directories found under {root}. "
            "Pass --dataset NAME=ROOT or use --source features/csv."
        )

    class_names = sorted(set(train_dirs) | set(test_dirs))
    counts = []
    for idx, class_name in enumerate(class_names):
        train_count = count_images_in_class_dir(train_dirs[class_name]) if class_name in train_dirs else 0
        test_count = count_images_in_class_dir(test_dirs[class_name]) if class_name in test_dirs else 0
        counts.append(ClassCount(spec.name, idx, class_name, float(train_count), float(test_count)))
    return counts


def torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalized_label_count(raw_count: int, factor: float) -> float:
    if factor <= 0:
        raise ValueError("Feature view factors must be positive.")
    value = raw_count / factor
    rounded = round(value)
    return float(rounded) if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-6) else value


def count_feature_payload(payload: dict, split: str, count_mode: str, view_factor: float) -> List[float]:
    import torch

    labels = payload["labels"].long()
    classes = payload.get("classes")
    if classes is None:
        max_label = int(labels.max().item()) if labels.numel() else -1
        classes = [f"class_{i:02d}" for i in range(max_label + 1)]

    if count_mode == "unique-paths" and "paths" in payload:
        per_class_paths = [set() for _ in classes]
        for label, path in zip(labels.tolist(), payload["paths"]):
            per_class_paths[int(label)].add(str(path))
        return [float(len(paths)) for paths in per_class_paths]

    if count_mode == "unique-paths":
        print(
            f"[warn] {split} feature cache has no paths; falling back to label counts "
            f"divided by view factor {view_factor:g}.",
            file=sys.stderr,
        )

    bincount = torch.bincount(labels, minlength=len(classes)).tolist()
    return [normalized_label_count(int(count), view_factor) for count in bincount]


def load_feature_counts(spec: DatasetSpec, args: argparse.Namespace) -> List[ClassCount]:
    if spec.config is None:
        raise ValueError(f"No feature config configured for {spec.name}")
    config = get_config(spec.config)
    train_payload = torch_load(Path(config["fine_train_path"]))
    test_payload = torch_load(Path(config["fine_test_path"]))

    classes = train_payload.get("classes") or test_payload.get("classes")
    if classes is None:
        max_label = max(
            int(train_payload["labels"].max().item()) if train_payload["labels"].numel() else -1,
            int(test_payload["labels"].max().item()) if test_payload["labels"].numel() else -1,
        )
        classes = [f"class_{i:02d}" for i in range(max_label + 1)]

    train_counts = count_feature_payload(
        train_payload,
        "train",
        args.feature_count_mode,
        args.feature_train_view_factor,
    )
    test_counts = count_feature_payload(
        test_payload,
        "test",
        args.feature_count_mode,
        args.feature_test_view_factor,
    )

    n_classes = max(len(classes), len(train_counts), len(test_counts))
    counts = []
    for idx in range(n_classes):
        class_name = classes[idx] if idx < len(classes) else f"class_{idx:02d}"
        train = train_counts[idx] if idx < len(train_counts) else 0.0
        test = test_counts[idx] if idx < len(test_counts) else 0.0
        counts.append(ClassCount(spec.name, idx, class_name, train, test))
    return counts


def load_csv_counts(path: Path) -> Tuple[List[str], Dict[str, List[ClassCount]]]:
    if path is None:
        raise ValueError("--csv is required when --source csv is used.")
    by_dataset: Dict[str, List[ClassCount]] = {}
    dataset_order: List[str] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"dataset", "class", "train", "test"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required CSV columns: {sorted(missing)}")

        next_index: Dict[str, int] = {}
        for row in reader:
            dataset = row["dataset"].strip()
            class_name = row["class"].strip()
            if not dataset or not class_name:
                continue
            if dataset not in by_dataset:
                by_dataset[dataset] = []
                dataset_order.append(dataset)
                next_index[dataset] = 0

            if row.get("class_index"):
                class_index = int(row["class_index"])
            else:
                class_index = next_index[dataset]
            next_index[dataset] = max(next_index[dataset], class_index + 1)

            by_dataset[dataset].append(
                ClassCount(
                    dataset=dataset,
                    class_index=class_index,
                    class_name=class_name,
                    train=float(row["train"]),
                    test=float(row["test"]),
                )
            )

    return dataset_order, by_dataset


def load_counts(args: argparse.Namespace) -> Tuple[List[str], Dict[str, List[ClassCount]]]:
    if args.source == "csv":
        return load_csv_counts(args.csv)

    if args.source == "raw" and args.dataset is None and args.download_missing:
        ensure_default_raw_datasets()

    specs = dataset_specs_from_args(args)
    by_dataset: Dict[str, List[ClassCount]] = {}
    for spec in specs:
        if args.source == "raw":
            by_dataset[spec.name] = load_raw_counts(spec)
        elif args.source == "features":
            by_dataset[spec.name] = load_feature_counts(spec, args)
        else:
            raise ValueError(f"Unsupported source: {args.source}")
    return [spec.name for spec in specs], by_dataset


def ensure_default_raw_datasets() -> None:
    from scripts.ensure_datasets import (
        configure_caches,
        dataset_ready,
        ensure_aqua20,
        ensure_fish4k,
        ensure_sea23,
    )

    download_dir = PROJECT_ROOT / "data" / "_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    configure_caches(download_dir)

    ensure_args = argparse.Namespace(
        fish4k_root=Path(FISH4K_ROOT),
        sea23_root=Path(SEA23_ROOT),
        aqua20_root=Path(AQUA20_ROOT),
        sea23_src=None,
        download_dir=download_dir,
        seed=42,
        fish4k_test_ratio=0.2,
        fish4k_min_test=5,
        sea23_test_ratio=0.2,
        sea23_symlink=False,
        force=False,
    )

    checks = [
        ("Fish4Knowledge", ensure_args.fish4k_root, ensure_fish4k),
        ("Sea Animals", ensure_args.sea23_root, ensure_sea23),
        ("AQUA20", ensure_args.aqua20_root, ensure_aqua20),
    ]
    for dataset_name, root, handler in checks:
        if dataset_ready(root):
            continue
        print(f"{dataset_name} is missing; preparing {root}")
        handler(ensure_args)


def short_name(value: str, max_len: int = 18) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "..."


def format_count(value: float) -> str:
    return f"{int(round(value))}" if math.isclose(value, round(value), abs_tol=1e-6) else f"{value:.1f}"


def write_summary_csv(
    path: Path,
    dataset_order: Sequence[str],
    by_dataset: Dict[str, List[ClassCount]],
    class_slot_order: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "plot_slot",
            "class_index",
            "class",
            "train",
            "test",
            "total",
            "train_pct",
            "test_pct",
        ])
        for dataset in dataset_order:
            by_index = {item.class_index: item for item in by_dataset.get(dataset, [])}
            for plot_slot, class_index in enumerate(class_slot_order):
                item = by_index.get(class_index)
                if item is None:
                    writer.writerow([dataset, plot_slot, class_index, "", 0, 0, 0, 0, 0])
                    continue
                writer.writerow([
                    item.dataset,
                    plot_slot,
                    item.class_index,
                    item.class_name,
                    format_count(item.train),
                    format_count(item.test),
                    format_count(item.total),
                    f"{item.train_pct:.4f}",
                    f"{item.test_pct:.4f}",
                ])


def make_x_labels(
    mode: str,
    dataset_order: Sequence[str],
    by_dataset: Dict[str, List[ClassCount]],
    class_slot_order: Sequence[int],
) -> List[str]:
    if mode == "index":
        return [f"C{class_index + 1:02d}" for class_index in class_slot_order]

    per_dataset = {
        dataset: {item.class_index: item.class_name for item in by_dataset.get(dataset, [])}
        for dataset in dataset_order
    }
    labels = []
    for class_index in class_slot_order:
        lines = [f"C{class_index + 1:02d}"]
        for dataset in dataset_order:
            ds_key = "".join(word[0] for word in dataset.split() if word)[:2].upper()
            name = per_dataset[dataset].get(class_index, "n/a")
            lines.append(f"{ds_key}: {short_name(name, 14)}")
        labels.append("\n".join(lines))
    return labels


def batch_sort_value(
    class_index: int,
    dataset_order: Sequence[str],
    by_dataset: Dict[str, List[ClassCount]],
    metric: str,
) -> float:
    totals = []
    for dataset in dataset_order:
        by_index = {item.class_index: item for item in by_dataset.get(dataset, [])}
        item = by_index.get(class_index)
        totals.append(item.total if item is not None else 0.0)
    if metric == "max":
        return max(totals, default=0.0)
    if metric in ("class-count", "aggregate"):
        return sum(totals)
    raise ValueError(f"Unsupported sort batch metric: {metric}")


def build_class_slot_order(
    args: argparse.Namespace,
    dataset_order: Sequence[str],
    by_dataset: Dict[str, List[ClassCount]],
    n_slots: int,
) -> List[int]:
    class_slot_order = list(range(n_slots))
    if not args.sort_batches:
        return class_slot_order

    return sorted(
        class_slot_order,
        key=lambda class_index: (
            -batch_sort_value(class_index, dataset_order, by_dataset, args.sort_batch_metric),
            class_index,
        ),
    )


def scale_count(value: float, mode: str) -> float:
    value = max(float(value), 0.0)
    if mode == "linear":
        return value
    if mode == "sqrt":
        return math.sqrt(value)
    if mode in ("log", "log1p"):
        return math.log1p(value)
    raise ValueError(f"Unsupported yscale: {mode}")


def format_tick_label(value: float) -> str:
    if value >= 1000 and math.isclose(value % 1000, 0.0, abs_tol=1e-6):
        return f"{int(value / 1000)}k"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return format_count(value)


def nice_linear_ticks(y_max: float, target_ticks: int = 6) -> List[float]:
    if y_max <= 0:
        return [0.0, 1.0]
    raw_step = y_max / max(target_ticks - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for factor in (1, 2, 5, 10):
        step = factor * magnitude
        if step >= raw_step:
            break
    top = math.ceil(y_max / step) * step
    ticks = []
    current = 0.0
    while current <= top + step * 0.5:
        ticks.append(current)
        current += step
    return ticks


def nonlinear_ticks(y_max: float) -> List[float]:
    if y_max <= 10:
        return [float(i) for i in range(0, int(math.ceil(y_max)) + 1)]

    ticks = [0.0, 1.0, 5.0]
    power = 1
    while 10 ** power <= y_max:
        for factor in (1, 5):
            value = float(factor * (10 ** power))
            if value <= y_max * 1.001:
                ticks.append(value)
        power += 1
    return sorted(set(ticks))


def configure_y_axis(ax, y_max: float, yscale: str) -> float:
    scaled_y_max = scale_count(y_max, yscale)
    if yscale == "linear":
        ticks = nice_linear_ticks(y_max)
    else:
        ticks = nonlinear_ticks(y_max)
    ax.set_yticks([scale_count(value, yscale) for value in ticks])
    ax.set_yticklabels([format_tick_label(value) for value in ticks])
    return scaled_y_max


def label_segment(
    ax,
    x: float,
    bottom: float,
    height: float,
    total: float,
    text: str,
    color: str,
    args: argparse.Namespace,
    y_max: float,
) -> None:
    if args.label_mode == "none" or height <= 0 or total <= 0:
        return
    if args.label_mode == "auto" and height < y_max * args.label_min_fraction:
        return
    ax.text(
        x,
        bottom + height / 2.0,
        text,
        ha="center",
        va="center",
        rotation=90,
        fontsize=7,
        color=color,
        clip_on=True,
    )


def configure_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "spymamba_matplotlib_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def plot_chart(
    args: argparse.Namespace,
    dataset_order: Sequence[str],
    by_dataset: Dict[str, List[ClassCount]],
    class_slot_order: Sequence[int],
) -> None:
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    n_slots = len(class_slot_order)
    x_positions = list(range(n_slots))
    n_datasets = len(dataset_order)
    offsets = [
        (i - (n_datasets - 1) / 2.0) * args.bar_width
        for i in range(n_datasets)
    ]
    y_max = max(
        (item.total for counts in by_dataset.values() for item in counts),
        default=1.0,
    )
    if y_max <= 0:
        y_max = 1.0
    scaled_y_max = scale_count(y_max, args.yscale)

    for ds_idx, dataset in enumerate(dataset_order):
        color = DEFAULT_COLORS[ds_idx % len(DEFAULT_COLORS)]
        by_index = {item.class_index: item for item in by_dataset.get(dataset, [])}
        for plot_slot, class_index in enumerate(class_slot_order):
            item = by_index.get(class_index)
            x = x_positions[plot_slot] + offsets[ds_idx]
            if item is None or item.total <= 0:
                ax.bar(
                    x,
                    0,
                    width=args.bar_width * 0.92,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=0.8,
                )
                if args.show_missing:
                    ax.text(
                        x,
                        0,
                        "n/a",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=6,
                        color=color,
                    )
                continue

            train_pct = f"{item.train_pct:.{args.percent_decimals}f}%"
            test_pct = f"{item.test_pct:.{args.percent_decimals}f}%"
            train_top = scale_count(item.train, args.yscale)
            total_top = scale_count(item.total, args.yscale)
            train_height = train_top
            test_height = max(total_top - train_top, 0.0)
            ax.bar(
                x,
                train_height,
                width=args.bar_width * 0.92,
                color=color,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.4,
            )
            ax.bar(
                x,
                test_height,
                bottom=train_top,
                width=args.bar_width * 0.92,
                color=color,
                alpha=0.36,
                edgecolor=color,
                linewidth=0.4,
                hatch="//",
            )
            label_segment(ax, x, 0.0, train_height, item.total, train_pct, "white", args, scaled_y_max)
            label_segment(ax, x, train_top, test_height, item.total, test_pct, "#202020", args, scaled_y_max)

    for boundary in [i + 0.5 for i in range(n_slots - 1)]:
        ax.axvline(boundary, color="#dddddd", linewidth=0.5, zorder=0)

    if args.class_labels:
        x_labels = make_x_labels(args.x_label_mode, dataset_order, by_dataset, class_slot_order)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            x_labels,
            rotation=0 if args.x_label_mode == "index" else 75,
            ha="center" if args.x_label_mode == "index" else "right",
            fontsize=8 if args.x_label_mode == "index" else 6,
        )
    else:
        ax.set_xticks(x_positions)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_xlabel(f"Class batches ({len(dataset_order)} dataset bars each)")
    ylabel = "Number of samples per class"
    if args.yscale != "linear":
        scale_label = "log1p" if args.yscale == "log" else args.yscale
        ylabel += f" ({scale_label} height scale)"
    ax.set_ylabel(ylabel)
    ax.set_title(args.title, pad=12)
    ax.set_xlim(-args.x_margin, n_slots - 1 + args.x_margin)
    scaled_y_max = configure_y_axis(ax, y_max, args.yscale)
    ax.set_ylim(0, scaled_y_max * (1.0 + args.y_padding))

    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)

    dataset_handles = [
        mpatches.Patch(color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)], label=dataset)
        for i, dataset in enumerate(dataset_order)
    ]
    split_handles = [
        mpatches.Patch(facecolor="#777777", alpha=0.82, label="Train segment"),
        mpatches.Patch(facecolor="#777777", alpha=0.36, hatch="//", label="Test segment"),
    ]
    ax.legend(
        handles=dataset_handles + split_handles,
        ncol=min(len(dataset_order) + 2, 5),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=9,
    )

    note = (
        f"{n_slots} class slots x {len(dataset_order)} datasets = "
        f"{n_slots * len(dataset_order)} bar positions"
    )
    if args.sort_batches:
        note += f" | sorted by {args.sort_batch_metric}"
    if args.yscale != "linear":
        scale_label = "log1p" if args.yscale == "log" else args.yscale
        note += f" | {scale_label} height scale"
    ax.text(
        1.0,
        -0.12 if args.x_label_mode == "index" else -0.28,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_order, by_dataset = load_counts(args)
    if not dataset_order:
        raise RuntimeError("No datasets were loaded.")

    max_classes = max((len(counts) for counts in by_dataset.values()), default=0)
    n_slots = args.max_class_slots or max_classes
    if n_slots <= 0:
        raise RuntimeError("No classes were found.")

    class_slot_order = build_class_slot_order(args, dataset_order, by_dataset, n_slots)
    plot_chart(args, dataset_order, by_dataset, class_slot_order)

    summary_csv = args.summary_csv
    if summary_csv is None and not args.no_summary_csv:
        summary_csv = args.output.with_suffix(".csv")
    if summary_csv is not None:
        write_summary_csv(summary_csv, dataset_order, by_dataset, class_slot_order)

    nonzero_bars = sum(
        1
        for dataset in dataset_order
        for item in by_dataset.get(dataset, [])
        if item.class_index < n_slots and item.total > 0
    )
    print(f"Saved figure: {args.output}")
    if summary_csv is not None:
        print(f"Saved summary: {summary_csv}")
    print(
        f"Rendered {n_slots * len(dataset_order)} bar positions "
        f"({nonzero_bars} non-empty) across {n_slots} class slots."
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
