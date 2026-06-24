#!/usr/bin/env python3
"""Generate a confusion matrix for the AQUA20 CLIP-feature classifier."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spiral_project.config import get_dataset_config  # noqa: E402
from spiral_project.data import ClipFeatureDataset  # noqa: E402
from spiral_project.trainer import build_feature_classifier, make_run_name  # noqa: E402
from spiral_project.utils import ensure_dir  # noqa: E402


def parse_args():
    config = get_dataset_config("aqua20_clip")
    default_checkpoint = PROJECT_ROOT / f"best_model_{make_run_name('aqua20_clip', 'clip', 42)}.pth"
    parser = argparse.ArgumentParser(
        description="Generate a confusion matrix for the AQUA20 CLIP feature dataset."
    )
    parser.add_argument(
        "--features-path",
        default=os.path.join(config["data_dir"], "test_features.pt"),
        help="Path to a CLIP feature .pt split file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(default_checkpoint),
        help="Path to the trained CLIP-feature classifier checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "confusion_matrices" / "aqua20_clip"),
        help="Directory where matrix files are saved.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Also save a row-normalized confusion matrix image.",
    )
    return parser.parse_args()


def compute_confusion_matrix(model, dataset, loader, device):
    num_classes = len(dataset.classes)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    correct = 0
    total = 0

    model.eval()
    with torch.inference_mode():
        for features, targets in tqdm(loader, desc="Predict"):
            features = features.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            outputs = model(features)
            predictions = outputs.argmax(dim=1)
            for true_label, predicted_label in zip(targets.cpu(), predictions.cpu()):
                matrix[int(true_label), int(predicted_label)] += 1
            correct += predictions.eq(targets).sum().item()
            total += targets.numel()

    return matrix, 100.0 * correct / total


def save_matrix_csv(matrix, classes, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *classes])
        for class_name, row in zip(classes, matrix.tolist()):
            writer.writerow([class_name, *row])


def save_matrix_plot(matrix, classes, output_path, title, normalize=False):
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped PNG confusion matrix.")
        return

    values = matrix.float()
    if normalize:
        values = values / values.sum(dim=1, keepdim=True).clamp_min(1)

    fig_size = max(8, len(classes) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(values.numpy(), cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx].item()
            if normalize:
                label = f"{value:.2f}" if value >= 0.01 else ""
            else:
                label = str(int(value)) if value > 0 else ""
            if label:
                color = "white" if value > values.max().item() * 0.55 else "black"
                ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=6, color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    config = get_dataset_config("aqua20_clip")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ClipFeatureDataset(args.features_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_feature_classifier(config, device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    matrix, accuracy = compute_confusion_matrix(model, dataset, loader, device)

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    torch.save(
        {
            "confusion_matrix": matrix,
            "classes": dataset.classes,
            "accuracy": accuracy,
            "features_path": args.features_path,
            "checkpoint": args.checkpoint,
        },
        out_dir / "confusion_matrix.pt",
    )
    save_matrix_csv(matrix, dataset.classes, out_dir / "confusion_matrix.csv")
    save_matrix_plot(
        matrix,
        dataset.classes,
        out_dir / "confusion_matrix.png",
        title=f"AQUA20 CLIP Confusion Matrix ({accuracy:.2f}% acc)",
    )
    if args.normalize:
        save_matrix_plot(
            matrix,
            dataset.classes,
            out_dir / "confusion_matrix_normalized.png",
            title="AQUA20 CLIP Confusion Matrix (row-normalized)",
            normalize=True,
        )

    with open(out_dir / "confusion_matrix_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "accuracy": accuracy,
                "features_path": args.features_path,
                "checkpoint": args.checkpoint,
                "num_samples": len(dataset),
                "num_classes": len(dataset.classes),
            },
            handle,
            indent=2,
        )
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Saved confusion matrix files to {out_dir}")


if __name__ == "__main__":
    main()
