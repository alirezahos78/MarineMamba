#!/usr/bin/env python3
"""
umap_vis.py — 6-panel UMAP visualisation for the AQUA20 train+test sets.

Train samples are drawn as small filled circles; test samples as larger ×
markers so you can see exactly where test points land relative to training
clusters.  Highlight classes (e.g. flatworm, rayfish) with --highlight to
make them opaque and everything else semi-transparent.

Panels:
  1. CLIP ViT-B/16  (mean-pooled 14×14 spatial grid, 768-d)
  2. CLIP ViT-B/32  (mean-pooled  7×7 spatial grid, 768-d)
  3. Mamba fine branch output   (128-d, post-SpyMamba B/16 branch)
  4. Mamba coarse branch output (128-d, post-SpyMamba B/32 branch)
  5. Mamba fine ⊕ coarse concatenated (256-d)
  6. Raw train images (flattened pixels, img_size×img_size×3)

Usage:
    cd SpyMamba/
    python3 scripts/umap_vis.py
    python3 scripts/umap_vis.py --highlight flatworm rayfish
    python3 scripts/umap_vis.py --no-mamba --highlight flatworm rayfish
    python3 scripts/umap_vis.py --n-neighbors 20 --min-dist 0.15
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm
from umap import UMAP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from spymamba.config import get_config
from spymamba.trainer import build_model
from spymamba.paths import (
    AQUA20_ROOT, SEA23_ROOT, FISH4K_ROOT,
    B16_FEATURES_ROOT,
    B32_FEATURES_ROOT,
    CLS_FEATURES_PATH,
    LOGS_DIR,
    PROJECT_ROOT as _ROOT,
)

_DATASET_ROOTS = {
    "aqua20_pyramid_hybrid_128_focal_balanced": AQUA20_ROOT,
    "sea23_pyramid_hybrid_128_focal_balanced":  SEA23_ROOT,
    "fish4k_baseline":                          FISH4K_ROOT,
}

ROOT = Path(_ROOT)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config",      default="aqua20_pyramid_hybrid_128_focal_balanced")
    p.add_argument("--ckpt",        default=None)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist",    type=float, default=0.1)
    p.add_argument("--pca-dim",     type=int, default=50,
                   help="PCA pre-reduction before UMAP for high-d features (0 = skip).")
    p.add_argument("--img-size",    type=int, default=64)
    p.add_argument("--batch-size",  type=int, default=512)
    p.add_argument("--no-mamba",    action="store_true",
                   help="Skip panels 3-5 (no checkpoint needed).")
    p.add_argument("--test-only",   action="store_true",
                   help="Fit and plot UMAP on the test set only (ignores training samples).")
    p.add_argument("--highlight",   nargs="+", default=None,
                   help="Class names to highlight (all others become faint). "
                        "Example: --highlight flatworm rayfish")
    p.add_argument("--output-dir",  default=None)
    return p.parse_args()


# ── UMAP helpers ───────────────────────────────────────────────────────────────

def run_umap(features, n_neighbors, min_dist, pca_dim, seed, metric="cosine"):
    X = features.float().numpy() if isinstance(features, torch.Tensor) else np.asarray(features, dtype=np.float32)
    if pca_dim > 0 and X.shape[1] > pca_dim:
        print(f"    PCA {X.shape[1]}-d → {pca_dim}-d ...", end=" ", flush=True)
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
        print("done.")
    print(f"    UMAP {X.shape} ({metric}) ...", end=" ", flush=True)
    emb = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=seed,
        verbose=False,
    ).fit_transform(X)
    print("done.")
    return emb


def draw_panel(ax, emb_tr, lbl_tr, emb_te, lbl_te, classes, title, highlight=None):
    """
    Draw one UMAP panel.
    - Train: small filled circles (·)
    - Test:  larger × markers
    If highlight is given, highlighted classes are fully opaque; others are very faint.
    """
    cmap  = plt.cm.get_cmap("tab20", len(classes))
    hi_set = set(highlight) if highlight else None

    def alpha_for(cls_name):
        if hi_set is None:
            return 0.30, 0.70          # train_alpha, test_alpha
        return (0.55, 0.90) if cls_name in hi_set else (0.04, 0.10)

    # Draw train (circles, small)
    for c, cls_name in enumerate(classes):
        mask = (lbl_tr == c)
        if not mask.any():
            continue
        tr_a, _ = alpha_for(cls_name)
        ax.scatter(emb_tr[mask, 0], emb_tr[mask, 1],
                   c=[cmap(c)], s=4, alpha=tr_a,
                   linewidths=0, rasterized=True, zorder=2)

    # Draw test (×, larger)
    if emb_te is not None:
        for c, cls_name in enumerate(classes):
            mask = (lbl_te == c)
            if not mask.any():
                continue
            _, te_a = alpha_for(cls_name)
            ax.scatter(emb_te[mask, 0], emb_te[mask, 1],
                       c=[cmap(c)], s=55, alpha=te_a,
                       marker="x", linewidths=1.2, rasterized=True, zorder=3)

    ax.set_title(title, fontsize=8.5, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


# ── Feature loaders ────────────────────────────────────────────────────────────

def load_clip_features(config, split):
    key_fine   = "fine_train_path"   if split == "train" else "fine_test_path"
    key_coarse = "coarse_train_path" if split == "train" else "coarse_test_path"
    b16 = torch.load(config[key_fine],   map_location="cpu", weights_only=True)
    b32 = torch.load(config[key_coarse], map_location="cpu", weights_only=True)
    b16_feat = b16["features"].float().flatten(2).mean(dim=2)   # [N, 768]
    b32_feat = b32["features"].float().flatten(2).mean(dim=2)
    labels   = b16["labels"].long().numpy()
    classes  = b16["classes"]
    return b16_feat, b32_feat, labels, classes


def extract_mamba_features(config, ckpt, device, batch_size, split):
    model = build_model(config, device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    key_fine   = "fine_train_path"   if split == "train" else "fine_test_path"
    key_coarse = "coarse_train_path" if split == "train" else "coarse_test_path"
    b16_pt = torch.load(config[key_fine],   map_location="cpu", weights_only=True)
    b32_pt = torch.load(config[key_coarse], map_location="cpu", weights_only=True)
    cls_all = torch.load(config["cls_path"], map_location="cpu", weights_only=True)[split]

    ff  = b16_pt["features"].float()
    cf  = b32_pt["features"].float()
    lbl = b16_pt["labels"].long()
    cls = b16_pt["classes"]

    n_unique = cls_all["ViT-B-16"].shape[0]
    idx      = torch.arange(ff.shape[0]) % n_unique
    fc_all   = cls_all["ViT-B-16"].float()[idx]
    cc_all   = cls_all["ViT-B-32"].float()[idx]

    nb = (device == "cuda")
    fine_outs, coarse_outs = [], []
    with torch.inference_mode():
        for s in tqdm(range(0, len(lbl), batch_size), desc=f"    fwd {split}", leave=False, file=sys.stderr):
            e = min(s + batch_size, len(lbl))
            fo = model.fine_branch(  ff[s:e].to(device, non_blocking=nb),
                                     fc_all[s:e].to(device, non_blocking=nb))
            co = model.coarse_branch(cf[s:e].to(device, non_blocking=nb),
                                     cc_all[s:e].to(device, non_blocking=nb))
            fine_outs.append(fo.cpu())
            coarse_outs.append(co.cpu())

    return (torch.cat(fine_outs), torch.cat(coarse_outs), lbl.numpy(), cls)


def load_raw_images(aqua20_root, classes, img_size, split="train"):
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    c2i = {c: i for i, c in enumerate(classes)}
    imgs, labels = [], []
    for cls_name in classes:
        cls_dir = os.path.join(aqua20_root, split, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            try:
                img = Image.open(os.path.join(cls_dir, fname)).convert("RGB")
                imgs.append(tf(img).flatten())
                labels.append(c2i[cls_name])
            except Exception:
                pass
    return torch.stack(imgs).numpy(), np.array(labels, dtype=np.int64)


def print_cosine_report(config, ckpt, device, classes, batch_size=512):
    """Print train vs test centroid cosine similarity for the two classes closest in test."""
    model = build_model(config, device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    centroids = {}
    for split in ("train", "test"):
        key_fine   = "fine_train_path"   if split == "train" else "fine_test_path"
        key_coarse = "coarse_train_path" if split == "train" else "coarse_test_path"
        b16 = torch.load(config[key_fine],   map_location="cpu", weights_only=True)
        b32 = torch.load(config[key_coarse], map_location="cpu", weights_only=True)
        cls_all = torch.load(config["cls_path"], map_location="cpu", weights_only=True)[split]
        ff, cf = b16["features"].float(), b32["features"].float()
        lbl = b16["labels"].long()
        n_u  = cls_all["ViT-B-16"].shape[0]
        idx  = torch.arange(ff.shape[0]) % n_u
        fc, cc = cls_all["ViT-B-16"].float()[idx], cls_all["ViT-B-32"].float()[idx]
        nb = (device == "cuda")
        feats = []
        with torch.inference_mode():
            for s in range(0, len(lbl), batch_size):
                e = min(s + batch_size, len(lbl))
                feats.append(model.get_features(ff[s:e].to(device, non_blocking=nb),
                                                cf[s:e].to(device, non_blocking=nb),
                                                fc[s:e].to(device, non_blocking=nb),
                                                cc[s:e].to(device, non_blocking=nb)).cpu())
        feats = torch.cat(feats)
        c_dict = {}
        for i, name in enumerate(classes):
            m = (lbl == i)
            if m.any():
                c_dict[name] = F.normalize(feats[m].mean(0), dim=0)
        centroids[split] = (c_dict, feats, lbl)

    print("\n── Cosine similarity: flatworm centroid → every class ─────────────────")
    for split in ("train", "test"):
        c_dict, feats, lbl = centroids[split]
        flat_c = c_dict.get("flatworm")
        if flat_c is None:
            continue
        print(f"\n  {split.upper()}:")
        sims = [(F.cosine_similarity(flat_c.unsqueeze(0), c_dict[n].unsqueeze(0)).item(), n)
                for n in classes if n in c_dict]
        for sim, name in sorted(sims, reverse=True)[:6]:
            tag = " ◄" if name == "rayfish" else ("  [self]" if name == "flatworm" else "")
            print(f"    {name:22s} {sim:+.4f}{tag}")

        flat_i = classes.index("flatworm")
        ray_i  = classes.index("rayfish")
        flat_f = F.normalize(feats[lbl == flat_i], dim=1)
        ray_f  = F.normalize(feats[lbl == ray_i],  dim=1)
        intra  = (flat_f @ flat_f.T).fill_diagonal_(0).sum() / max(len(flat_f)*(len(flat_f)-1), 1)
        cross  = (flat_f @ ray_f.T).mean()
        print(f"    flatworm intra-class cosine : {intra:.4f}")
        print(f"    flatworm↔rayfish cross cosine: {cross:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = get_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = args.ckpt or str(ROOT / f"best_model_{args.config}_seed_{args.seed}.pth")
    out_dir = Path(args.output_dir) if args.output_dir else Path(LOGS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_only = args.test_only
    split_label = "test only" if test_only else "train + test"
    print("=" * 65)
    print(f"UMAP Visualisation — {args.config}  [{split_label}]")
    print(f"  Config    : {args.config}")
    print(f"  Device    : {device}")
    print(f"  UMAP      : n_neighbors={args.n_neighbors}  min_dist={args.min_dist}")
    print(f"  Highlight : {args.highlight or 'all (equal)'}")
    print("=" * 65)

    # ── CLIP features ─────────────────────────────────────────────────────────
    print("\n[1/2] CLIP B/16 & B/32 spatial features...")
    b16_te, b32_te, lbl_te, classes = load_clip_features(config, "test")
    N_te = len(lbl_te)

    if test_only:
        b16_fit, b32_fit = b16_te, b32_te
        lbl_tr = np.empty(0, dtype=np.int64)
        N_tr = 0
    else:
        b16_tr, b32_tr, lbl_tr, _ = load_clip_features(config, "train")
        N_tr = len(lbl_tr)
        b16_fit = torch.cat([b16_tr, b16_te])
        b32_fit = torch.cat([b32_tr, b32_te])

    print(f"  train={N_tr}  test={N_te}  classes={len(classes)}")

    print("  Panel 1 — CLIP B/16:")
    emb_b16 = run_umap(b16_fit, args.n_neighbors, args.min_dist, args.pca_dim, args.seed)
    if test_only:
        emb_b16_tr, emb_b16_te = None, emb_b16
    else:
        emb_b16_tr, emb_b16_te = emb_b16[:N_tr], emb_b16[N_tr:]

    print("  Panel 2 — CLIP B/32:")
    emb_b32 = run_umap(b32_fit, args.n_neighbors, args.min_dist, args.pca_dim, args.seed)
    if test_only:
        emb_b32_tr, emb_b32_te = None, emb_b32
    else:
        emb_b32_tr, emb_b32_te = emb_b32[:N_tr], emb_b32[N_tr:]

    # ── Mamba features ────────────────────────────────────────────────────────
    emb_fine_tr  = emb_fine_te  = None
    emb_coarse_tr= emb_coarse_te= None
    emb_cat_tr   = emb_cat_te   = None
    mamba_lbl_tr = lbl_tr
    mamba_lbl_te = lbl_te

    if args.no_mamba:
        print("\n[3/4/5] Mamba panels skipped (--no-mamba).")
    elif not Path(ckpt).exists():
        print(f"\n[3/4/5] Checkpoint not found: {ckpt} — skipping Mamba panels.")
    else:
        print(f"\n[3/4/5] Mamba branch features  ({Path(ckpt).name})...")
        if not test_only:
            print("  Extracting cosine similarity report...")
            print_cosine_report(config, ckpt, device, classes)

        fo_te, co_te, mamba_lbl_te, _ = extract_mamba_features(
            config, ckpt, device, args.batch_size, "test")
        cat_te = torch.cat([fo_te, co_te], dim=-1)

        if test_only:
            fo_fit, co_fit, cat_fit = fo_te, co_te, cat_te
        else:
            fo_tr, co_tr, mamba_lbl_tr, _ = extract_mamba_features(
                config, ckpt, device, args.batch_size, "train")
            cat_tr  = torch.cat([fo_tr, co_tr], dim=-1)
            fo_fit  = torch.cat([fo_tr,  fo_te])
            co_fit  = torch.cat([co_tr,  co_te])
            cat_fit = torch.cat([cat_tr, cat_te])

        print("  Panel 3 — Mamba fine branch (128-d):")
        emb_fo = run_umap(fo_fit, args.n_neighbors, args.min_dist, 0, args.seed, "euclidean")
        if test_only:
            emb_fine_tr, emb_fine_te = None, emb_fo
        else:
            emb_fine_tr = emb_fo[:len(mamba_lbl_tr)]
            emb_fine_te = emb_fo[len(mamba_lbl_tr):]

        print("  Panel 4 — Mamba coarse branch (128-d):")
        emb_co = run_umap(co_fit, args.n_neighbors, args.min_dist, 0, args.seed, "euclidean")
        if test_only:
            emb_coarse_tr, emb_coarse_te = None, emb_co
        else:
            emb_coarse_tr = emb_co[:len(mamba_lbl_tr)]
            emb_coarse_te = emb_co[len(mamba_lbl_tr):]

        print("  Panel 5 — Mamba fine ⊕ coarse (256-d):")
        emb_cat = run_umap(cat_fit, args.n_neighbors, args.min_dist, 0, args.seed, "euclidean")
        if test_only:
            emb_cat_tr, emb_cat_te = None, emb_cat
        else:
            emb_cat_tr = emb_cat[:len(mamba_lbl_tr)]
            emb_cat_te = emb_cat[len(mamba_lbl_tr):]

    # ── Raw images ────────────────────────────────────────────────────────────
    print(f"\n[6] Raw images ({args.img_size}×{args.img_size})...")
    img_root = _DATASET_ROOTS.get(args.config, AQUA20_ROOT)
    img_te, img_lbl_te = load_raw_images(img_root, classes, args.img_size, "test")
    if test_only:
        img_fit, img_lbl_tr, emb_img_tr = img_te, np.empty(0, dtype=np.int64), None
    else:
        img_tr, img_lbl_tr = load_raw_images(img_root, classes, args.img_size, "train")
        img_fit = np.concatenate([img_tr, img_te])
    print(f"  test={len(img_lbl_te)}  dim={img_te.shape[1]}")
    print("  Panel 6 — Raw images:")
    emb_img = run_umap(img_fit, args.n_neighbors, args.min_dist, args.pca_dim, args.seed, "euclidean")
    if test_only:
        emb_img_te = emb_img
    else:
        emb_img_tr = emb_img[:len(img_lbl_tr)]
        emb_img_te = emb_img[len(img_lbl_tr):]

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nRendering figure...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.patch.set_facecolor("#f8f8f8")
    cmap = plt.cm.get_cmap("tab20", len(classes))

    def _n(n): return f"{n:,}"

    te_marker = "·" if test_only else "×"
    panel_defs = [
        (axes[0,0], emb_b16_tr,    lbl_tr,         emb_b16_te,    lbl_te,
         f"1. CLIP ViT-B/16  (mean-pool, 768-d)\n{_n(N_te)} test {te_marker}"),
        (axes[0,1], emb_b32_tr,    lbl_tr,         emb_b32_te,    lbl_te,
         f"2. CLIP ViT-B/32  (mean-pool, 768-d)\n{_n(N_te)} test {te_marker}"),
        (axes[0,2], emb_fine_tr,   mamba_lbl_tr,   emb_fine_te,   mamba_lbl_te,
         f"3. Mamba fine branch  (128-d)\n{_n(N_te)} test {te_marker}"),
        (axes[1,0], emb_coarse_tr, mamba_lbl_tr,   emb_coarse_te, mamba_lbl_te,
         f"4. Mamba coarse branch  (128-d)\n{_n(N_te)} test {te_marker}"),
        (axes[1,1], emb_cat_tr,    mamba_lbl_tr,   emb_cat_te,    mamba_lbl_te,
         f"5. Mamba fine ⊕ coarse  (256-d)\n{_n(N_te)} test {te_marker}"),
        (axes[1,2], emb_img_tr,    img_lbl_tr,     emb_img_te,    img_lbl_te,
         f"6. Raw images  ({args.img_size}×{args.img_size}×3 px)\n{_n(len(img_lbl_te))} test {te_marker}"),
    ]

    for ax, emb_tr_, lbl_tr_, emb_te_, lbl_te_, title in panel_defs:
        ax.set_facecolor("#efefef")
        # In test-only mode pass None for train so draw_panel skips it
        tr_arg = None if test_only else emb_tr_
        if emb_te_ is None and tr_arg is None:
            ax.text(0.5, 0.5, "not computed\n(checkpoint missing or --no-mamba)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#999", fontsize=10)
            ax.set_title(title, fontsize=8.5, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            draw_panel(ax, tr_arg, lbl_tr_, emb_te_, lbl_te_, classes, title, args.highlight)

    # Legend
    patches = [mpatches.Patch(color=cmap(i), label=classes[i]) for i in range(len(classes))]
    legend_handles = patches[:]
    if not test_only:
        legend_handles += [
            mlines.Line2D([], [], color="grey", marker="o", linestyle="None",
                          markersize=5, label="train (·)"),
            mlines.Line2D([], [], color="grey", marker="x", linestyle="None",
                          markersize=7, markeredgewidth=1.3, label="test (×)"),
        ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=11, fontsize=8.2,
               framealpha=0.9, edgecolor="#ccc", bbox_to_anchor=(0.5, -0.03))

    hi_note = f"  highlighted: {args.highlight}" if args.highlight else ""
    title_str = f"UMAP — {args.config}  [{split_label}]    " \
                f"n_neighbors={args.n_neighbors}  min_dist={args.min_dist}  " \
                f"PCA-pre={args.pca_dim if args.pca_dim else 'off'}{hi_note}"
    fig.suptitle(title_str, fontsize=10.5, y=1.01)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    ts = int(time.time())
    hi_tag   = ("_hi_" + "_".join(args.highlight)) if args.highlight else ""
    split_tag = "_test" if test_only else ""
    out_path = out_dir / f"umap_{args.config}{split_tag}{hi_tag}_{ts}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
