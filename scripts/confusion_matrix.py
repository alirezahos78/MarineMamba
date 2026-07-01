#!/usr/bin/env python3
"""
confusion_matrix.py — Print and save the confusion matrix for a trained SpyMamba model.

Usage:
    cd SpyMamba/
    python3 scripts/confusion_matrix.py
    python3 scripts/confusion_matrix.py --config aqua20_pyramid_hybrid_128_mlp
    python3 scripts/confusion_matrix.py --save-png
"""
import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from spymamba.config import get_config
from spymamba.paths import CLS_FEATURES_PATH
from spymamba.trainer import build_model


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default="aqua20_pyramid_hybrid_128_focal_balanced")
    p.add_argument("--ckpt",   default=None,
                   help="Checkpoint path; defaults to best_{config}_seed_{seed}.pth")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--save-png", dest="save_png", action="store_true",
                   help="Save a colour-coded PNG heatmap to logs/")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_config(args.config)
    model  = build_model(config, device)
    ckpt   = args.ckpt or str(
        PROJECT_ROOT / f"best_model_{args.config}_seed_{args.seed}.pth")
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    # ── Load test features ───────────────────────────────────────────────────
    fine_data   = torch.load(config["fine_test_path"],   map_location="cpu")
    coarse_data = torch.load(config["coarse_test_path"], map_location="cpu")
    cls_raw     = torch.load(CLS_FEATURES_PATH,          map_location="cpu")["test"]

    fine_feats     = fine_data["features"].float()
    coarse_feats   = coarse_data["features"].float()
    fine_cls_all   = cls_raw["ViT-B-16"].float()
    coarse_cls_all = cls_raw["ViT-B-32"].float()
    labels         = fine_data["labels"]
    classes        = fine_data["classes"]
    n, C           = len(labels), len(classes)

    # ── Run inference ────────────────────────────────────────────────────────
    preds = []
    bs    = 128
    for start in range(0, n, bs):
        sl = slice(start, start + bs)
        with torch.inference_mode():
            out = model(
                fine_feats[sl].to(device),
                coarse_feats[sl].to(device),
                fine_cls_all[sl].to(device),
                coarse_cls_all[sl].to(device),
            )
        preds.append(out.topk(3, dim=1).indices.cpu())
    all_topk = torch.cat(preds)          # [N, 3]
    preds    = all_topk[:, 0]            # top-1 predictions

    labels   = labels.long()

    # ── Build confusion matrix ───────────────────────────────────────────────
    cm = torch.zeros(C, C, dtype=torch.long)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    acc = preds.eq(labels).float().mean().item() * 100

    # ── Print to terminal ────────────────────────────────────────────────────
    col_w  = max(len(c) for c in classes) + 1
    num_w  = 5
    indent = " " * (col_w + 2)

    print(f"\n{'=' * 70}")
    print(f"Confusion Matrix  —  {args.config}  |  Top-1 accuracy: {acc:.2f}%")
    print(f"{'=' * 70}")

    # Header row (predicted labels)
    header = indent + "".join(f"{c[:num_w]:>{num_w}}" for c in classes)
    print(header)
    print(indent + "-" * (num_w * C))

    for i, row_cls in enumerate(classes):
        row_str = f"{row_cls:<{col_w}} |"
        for j in range(C):
            val = int(cm[i, j])
            if i == j:
                row_str += f"\033[92m{val:>{num_w}}\033[0m"   # green diagonal
            elif val > 0:
                row_str += f"\033[91m{val:>{num_w}}\033[0m"   # red off-diagonal
            else:
                row_str += f"{'':>{num_w}}"
        n_true = int(cm[i].sum())
        recall = int(cm[i, i]) / n_true * 100 if n_true > 0 else 0.0
        row_str += f"  | n={n_true:3d}  recall={recall:5.1f}%"
        print(row_str)

    print(indent + "-" * (num_w * C))

    # Precision row
    prec_str = f"{'precision':<{col_w}} |"
    for j in range(C):
        n_pred = int(cm[:, j].sum())
        prec   = int(cm[j, j]) / n_pred * 100 if n_pred > 0 else 0.0
        prec_str += f"{prec:>{num_w}.0f}"
    print(prec_str)
    print()

    # ── Per-class F1 summary ─────────────────────────────────────────────────
    print(f"{'Class':22s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}  "
          f"{'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Support':>7s}")
    print("-" * 72)
    macro_f1 = 0.0
    for i, cls in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        macro_f1 += f1
        print(f"{cls:22s}  {tp:4d}  {fp:4d}  {fn:4d}  "
              f"{prec:5.1f}%  {rec:5.1f}%  {f1:5.1f}%  {int(cm[i].sum()):7d}")
    macro_f1 /= C
    print("-" * 72)
    print(f"{'Macro avg':22s}  {'':4s}  {'':4s}  {'':4s}  "
          f"{'':6s}  {'':6s}  {macro_f1:5.1f}%")

    lbl_col = labels.unsqueeze(1)
    top1_correct = preds.eq(labels).sum().item()
    top2_correct = all_topk[:, :2].eq(lbl_col).any(dim=1).sum().item()
    top3_correct = all_topk[:, :3].eq(lbl_col).any(dim=1).sum().item()
    print(f"\nTop-1 accuracy : {top1_correct/n*100:.4f}%  ({top1_correct}/{n})")
    print(f"Top-2 accuracy : {top2_correct/n*100:.4f}%  ({top2_correct}/{n})")
    print(f"Top-3 accuracy : {top3_correct/n*100:.4f}%  ({top3_correct}/{n})")

    log_path = PROJECT_ROOT / "logs" / f"log_{args.config}_seed_{args.seed}.txt"
    with open(log_path, "a") as f:
        f.write(f"\n--- Evaluation summary (confusion_matrix.py) ---\n")
        f.write(f"Top-1 accuracy : {top1_correct/n*100:.4f}%  ({top1_correct}/{n})\n")
        f.write(f"Top-2 accuracy : {top2_correct/n*100:.4f}%  ({top2_correct}/{n})\n")
        f.write(f"Top-3 accuracy : {top3_correct/n*100:.4f}%  ({top3_correct}/{n})\n")
        f.write(f"Macro-F1       : {macro_f1:.2f}%\n")
    print(f"Appended to {log_path}")

    # ── Most confused pairs ──────────────────────────────────────────────────
    cm_off = cm.clone()
    cm_off.fill_diagonal_(0)
    vals, flat_idx = cm_off.view(-1).topk(10)
    print(f"\nTop confused pairs (predicted as → true):")
    for v, idx in zip(vals.tolist(), flat_idx.tolist()):
        if v == 0:
            break
        t, p = divmod(idx, C)
        print(f"  True={classes[t]:15s} → Pred={classes[p]:15s}  count={v}")

    # ── Optional PNG heatmap ─────────────────────────────────────────────────
    if args.save_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            import seaborn as sns

            cm_np   = cm.numpy().astype(float)
            cm_norm = cm_np / cm_np.sum(axis=1, keepdims=True).clip(min=1)

            # Annotation: show raw counts; blank cells that are zero
            annot = np.where(cm_np > 0, cm_np.astype(int).astype(str), "")

            fig, ax = plt.subplots(figsize=(16, 13))
            sns.heatmap(
                cm_norm,
                annot=annot,
                fmt="",
                cmap="Blues",
                vmin=0, vmax=1,
                linewidths=0.4,
                linecolor="white",
                xticklabels=classes,
                yticklabels=classes,
                cbar_kws={"label": "Recall (row-normalised)", "shrink": 0.8},
                ax=ax,
            )
            ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
            ax.set_ylabel("True",      fontsize=12, labelpad=10)
            ax.set_title(
                f"Confusion Matrix — {args.config}\n"
                f"Top-1 acc: {acc:.2f}%  |  Macro-F1: {macro_f1:.1f}%",
                fontsize=13, pad=14,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=9)

            plt.tight_layout()
            out_path = PROJECT_ROOT / "logs" / f"confusion_matrix_{args.config}.png"
            out_path.parent.mkdir(exist_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\nPNG saved: {out_path}")
        except ImportError as e:
            print(f"\n[warn] missing dependency ({e}) — skipping PNG")


if __name__ == "__main__":
    main()
