#!/usr/bin/env python3
"""
Create a 23-cluster stacked bar chart of train/test samples per class.

Each cluster contains three dataset bars in this default left-to-right order:
    sea_animals (left), fish4knowledge (middle), aqua20 (right)

Train is drawn at the bottom; test is drawn on top with a lighter shade of the
dataset color. Shorter datasets are padded with zero bars so every cluster has
three dataset bars. AQUA20 has 20 classes, so it is padded to 23 by default.

Usage:
    cd SpyMamba/
    python3 scripts/dataset_clustered_cumulative_barchart.py
    python3 scripts/dataset_clustered_cumulative_barchart.py --yscale linear
    python3 scripts/dataset_clustered_cumulative_barchart.py --sort-bars none
    python3 scripts/dataset_clustered_cumulative_barchart.py --connect-dataset-lines
"""
import argparse
import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from spymamba.paths import AQUA20_ROOT, FISH4K_ROOT, LOGS_DIR, SEA23_ROOT

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    root: Path
    color: str


@dataclass
class ClassCount:
    class_name: str
    train: int
    test: int
    padded: bool = False

    @property
    def total(self) -> int:
        return self.train + self.test


DATASETS = {
    "sea_animals": DatasetSpec("sea_animals", "Sea Animals", Path(SEA23_ROOT), "#2ca02c"),
    "fish4knowledge": DatasetSpec("fish4knowledge", "Fish4Knowledge", Path(FISH4K_ROOT), "#1f77b4"),
    "aqua20": DatasetSpec("aqua20", "AQUA20", Path(AQUA20_ROOT), "#d62728"),
}

DEFAULT_ORDER = ["sea_animals", "fish4knowledge", "aqua20"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="23-cluster stacked train/test bar chart with three dataset bars per cluster.",
    )
    parser.add_argument(
        "--order",
        nargs="+",
        choices=tuple(DATASETS),
        default=DEFAULT_ORDER,
        help="Dataset bar order inside each cluster from left to right.",
    )
    parser.add_argument(
        "--pad-to",
        type=int,
        default=None,
        help="Pad each dataset to this many clusters. Defaults to the maximum class count.",
    )
    parser.add_argument(
        "--sort-bars",
        choices=("total", "train", "test", "none"),
        default="total",
        help="Sort class bars inside each dataset descending by this count before clustering by rank.",
    )
    parser.add_argument(
        "--yscale",
        choices=("linear", "sqrt", "log1p"),
        default="linear",
        help=(
            "Height transform. linear keeps equal visual y-grid sections; "
            "sqrt/log1p make low-count classes more visible."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(LOGS_DIR) / "dataset_clustered_cumulative_barchart.png",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional counts CSV. Defaults to the output path with .csv suffix.",
    )
    parser.add_argument("--fig-width", type=float, default=15.0)
    parser.add_argument("--fig-height", type=float, default=7.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--bar-width", type=float, default=0.82)
    parser.add_argument("--cluster-gap", type=float, default=0.35)
    parser.add_argument(
        "--connect-dataset-lines",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Connect the top of each dataset's bars across clusters with one colored line per dataset.",
    )
    parser.add_argument("--y-padding", type=float, default=0.05)
    parser.add_argument("--title-font-size", type=float, default=18.0)
    parser.add_argument("--axis-label-font-size", type=float, default=16.0)
    parser.add_argument("--tick-font-size", type=float, default=14.0)
    parser.add_argument("--legend-font-size", type=float, default=14.0)
    parser.add_argument(
        "--uniform-y-ticks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use evenly increasing y-axis tick values, e.g. 0, 2k, 4k, ...",
    )
    parser.add_argument(
        "--y-tick-count",
        type=int,
        default=8,
        help="Target number of y-axis ticks when --uniform-y-ticks is enabled.",
    )
    parser.add_argument(
        "--y-tick-step",
        type=float,
        default=None,
        help="Optional exact y-axis tick interval in sample counts, e.g. 2000.",
    )
    parser.add_argument(
        "--title",
        default="Train/test samples per class rank clustered across datasets",
    )
    return parser.parse_args()


def configure_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "spymamba_matplotlib_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def count_images(class_dir: Path) -> int:
    return sum(
        1
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def split_class_dirs(root: Path, split: str) -> Dict[str, Path]:
    split_dir = root / split
    if not split_dir.is_dir():
        return {}
    return {item.name: item for item in sorted(split_dir.iterdir()) if item.is_dir()}


def load_counts(spec: DatasetSpec) -> List[ClassCount]:
    train_dirs = split_class_dirs(spec.root, "train")
    test_dirs = split_class_dirs(spec.root, "test")
    if not train_dirs and not test_dirs:
        raise RuntimeError(f"No train/test class folders found for {spec.label}: {spec.root}")

    class_names = sorted(set(train_dirs) | set(test_dirs))
    counts = []
    for class_name in class_names:
        train = count_images(train_dirs[class_name]) if class_name in train_dirs else 0
        test = count_images(test_dirs[class_name]) if class_name in test_dirs else 0
        counts.append(ClassCount(class_name, train, test))
    return counts


def sort_counts(counts: List[ClassCount], mode: str) -> List[ClassCount]:
    if mode == "none":
        return counts
    key_fn = {
        "total": lambda item: item.total,
        "train": lambda item: item.train,
        "test": lambda item: item.test,
    }[mode]
    return sorted(counts, key=lambda item: (-key_fn(item), item.class_name))


def pad_counts(counts: List[ClassCount], length: int) -> List[ClassCount]:
    padded = list(counts)
    for i in range(max(length - len(padded), 0)):
        padded.append(ClassCount(f"pad_{i + 1}", 0, 0, padded=True))
    return padded


def scale_count(value: float, mode: str) -> float:
    value = max(float(value), 0.0)
    if mode == "linear":
        return value
    if mode == "sqrt":
        return math.sqrt(value)
    if mode == "log1p":
        return math.log1p(value)
    raise ValueError(f"Unsupported yscale: {mode}")


def format_count(value: float) -> str:
    if value >= 1000:
        scaled = value / 1000
        if math.isclose(scaled, round(scaled), abs_tol=1e-6):
            return f"{int(round(scaled))}k"
        return f"{scaled:.1f}k"
    return f"{int(round(value))}"


def nonlinear_ticks(y_max: float) -> List[float]:
    if y_max <= 10:
        return [float(i) for i in range(int(math.ceil(y_max)) + 1)]

    ticks = [0.0, 1.0, 5.0]
    power = 1
    while 10 ** power <= y_max:
        for factor in (1, 5):
            value = float(factor * (10 ** power))
            if value <= y_max * 1.001:
                ticks.append(value)
        power += 1
    return sorted(set(ticks))


def linear_ticks(
    y_max: float,
    target_ticks: int = 8,
    tick_step: float | None = None,
) -> List[float]:
    if y_max <= 0:
        return [0.0, 1.0]
    if tick_step is not None:
        if tick_step <= 0:
            raise ValueError("--y-tick-step must be positive.")
        step = tick_step
    else:
        raw_step = y_max / max(target_ticks - 1, 1)
        magnitude = 10 ** math.floor(math.log10(raw_step))
        step = magnitude
        for factor in (1, 2, 2.5, 5, 10):
            candidate = factor * magnitude
            if candidate >= raw_step:
                step = candidate
                break
    top = math.ceil(y_max / step) * step
    ticks = []
    value = 0.0
    while value <= top + step * 0.5:
        ticks.append(value)
        value += step
    return ticks


def lighten(hex_color: str, amount: float = 0.48) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = round(r + (255 - r) * amount)
    g = round(g + (255 - g) * amount)
    b = round(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_summary_csv(
    path: Path,
    ordered_specs: Sequence[DatasetSpec],
    by_dataset: Dict[str, List[ClassCount]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_index", "dataset", "class", "train", "test", "total", "padded"])
        n_clusters = max(len(counts) for counts in by_dataset.values())
        for cluster_index in range(n_clusters):
            for spec in ordered_specs:
                item = by_dataset[spec.key][cluster_index]
                writer.writerow([
                    cluster_index,
                    spec.key,
                    "" if item.padded else item.class_name,
                    item.train,
                    item.test,
                    item.total,
                    int(item.padded),
                ])


def plot_chart(
    args: argparse.Namespace,
    ordered_specs: Sequence[DatasetSpec],
    by_dataset: Dict[str, List[ClassCount]],
) -> None:
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    max_total = max(
        (item.total for counts in by_dataset.values() for item in counts),
        default=1,
    )
    scaled_max = scale_count(max_total, args.yscale)
    n_clusters = max(len(counts) for counts in by_dataset.values())
    n_datasets = len(ordered_specs)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    cluster_centers = []
    line_points = {spec.key: {"x": [], "y": []} for spec in ordered_specs}
    cluster_width = n_datasets * args.bar_width
    for cluster_index in range(n_clusters):
        base_x = cluster_index * (cluster_width + args.cluster_gap)
        cluster_xs = []
        for dataset_index, spec in enumerate(ordered_specs):
            item = by_dataset[spec.key][cluster_index]
            x = base_x + dataset_index * args.bar_width
            train_color = spec.color
            test_color = lighten(spec.color)
            train_top = scale_count(item.train, args.yscale)
            total_top = scale_count(item.total, args.yscale)
            test_height = max(total_top - train_top, 0.0)
            alpha = 0.25 if item.padded else 1.0

            ax.bar(
                x,
                train_top,
                width=args.bar_width,
                color=train_color,
                alpha=alpha,
                linewidth=0,
            )
            ax.bar(
                x,
                test_height,
                bottom=train_top,
                width=args.bar_width,
                color=test_color,
                alpha=alpha,
                linewidth=0,
            )
            cluster_xs.append(x)
            line_points[spec.key]["x"].append(x)
            line_points[spec.key]["y"].append(total_top)
        cluster_centers.append((cluster_xs[0] + cluster_xs[-1]) / 2.0)

    if args.connect_dataset_lines:
        for spec in ordered_specs:
            points = line_points[spec.key]
            ax.plot(
                points["x"],
                points["y"],
                color=spec.color,
                linewidth=1.5,
                marker="o",
                markersize=2.2,
                zorder=5,
            )

    if args.uniform_y_ticks:
        ticks = linear_ticks(max_total, args.y_tick_count, args.y_tick_step)
    else:
        ticks = linear_ticks(max_total) if args.yscale == "linear" else nonlinear_ticks(max_total)
    ax.set_yticks([scale_count(value, args.yscale) for value in ticks])
    ax.set_yticklabels([format_count(value) for value in ticks])
    ax.tick_params(axis="y", labelsize=args.tick_font_size)
    y_limit_value = max(max_total, ticks[-1] if ticks else max_total)
    ax.set_ylim(0, scale_count(y_limit_value, args.yscale) * (1.0 + args.y_padding))

    ax.set_xticks(cluster_centers)
    ax.set_xticklabels([str(i + 1) for i in range(n_clusters)])
    ax.tick_params(axis="x", labelsize=args.tick_font_size)
    ax.set_xlim(
        -args.bar_width,
        (n_clusters - 1) * (cluster_width + args.cluster_gap) + cluster_width,
    )
    ax.set_xlabel(
        f"Class-rank clusters ({n_datasets} dataset bars each)",
        fontsize=args.axis_label_font_size,
    )

    ylabel = "Samples"
    if args.yscale != "linear":
        ylabel += f" ({args.yscale} scale)"
    ax.set_ylabel(ylabel, fontsize=args.axis_label_font_size)
    ax.set_title(args.title, pad=12, fontsize=args.title_font_size)

    for boundary in [
        (cluster_centers[i] + cluster_centers[i + 1]) / 2.0
        for i in range(len(cluster_centers) - 1)
    ]:
        ax.axvline(boundary, color="#dddddd", linewidth=0.7, zorder=0)

    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)

    handles = []
    for spec in ordered_specs:
        handles.append(mpatches.Patch(color=spec.color, label=f"{spec.label} train"))
        handles.append(mpatches.Patch(color=lighten(spec.color), label=f"{spec.label} test"))
    ax.legend(
        handles=handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=args.legend_font_size,
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ordered_specs = [DATASETS[key] for key in args.order]

    raw_counts = {
        spec.key: sort_counts(load_counts(spec), args.sort_bars)
        for spec in ordered_specs
    }
    pad_to = args.pad_to or max(len(counts) for counts in raw_counts.values())
    by_dataset = {
        key: pad_counts(counts, pad_to)
        for key, counts in raw_counts.items()
    }

    plot_chart(args, ordered_specs, by_dataset)

    summary_csv = args.summary_csv or args.output.with_suffix(".csv")
    write_summary_csv(summary_csv, ordered_specs, by_dataset)

    print(f"Saved figure: {args.output}")
    print(f"Saved summary: {summary_csv}")
    print(f"Dataset bar order inside each cluster: {', '.join(args.order)}")
    print(f"Clusters: {pad_to}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
