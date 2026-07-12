#!/usr/bin/env python3
"""
paper_metrics.py — Four sets of numbers for the paper.

Task 1 : Parameter breakdown + Mamba hyperparams
Task 2 : Macro-F1 from existing checkpoints (AQUA20 + Sea23, 11 seeds each)
Task 3 : Frozen CLIP baselines on AQUA20 (seeds 0,1,2)
Task 4 : Fish4K per-class analysis — MarineMamba only (Vim checkpoints were never saved)

Run from project root:
    python3 scripts/paper_metrics.py

JSON output → results/
"""

import copy
import json
import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from marinemamba.config import get_config
from marinemamba.data import DualCLIPFeatureDataset, build_dataloaders
from marinemamba.losses import FocalLoss, build_criterion
from marinemamba.model import MarineMamba, MarineMambaBlock, PyramidBranch, SCAtt
from marinemamba.paths import PROJECT_ROOT
from marinemamba.trainer import build_model
from marinemamba.utils import set_seed

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS_11 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
SEEDS_3  = [0, 1, 2]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dump(name, obj):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  → saved {path}")


def _heading(text):
    bar = "=" * 70
    print(f"\n{bar}\n{text}\n{bar}")


def _load_marinemamba_ckpt(ckpt_path, config_name, device):
    """Build model from config, load checkpoint by path."""
    cfg   = get_config(config_name)
    model = build_model(cfg, device)
    sd    = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Parameter breakdown
# ─────────────────────────────────────────────────────────────────────────────

class _FFNBlock(nn.Module):
    """Token-wise FFN + CBAM — identical to ablation.py build_no_mamba."""
    def __init__(self, dim, height, width, ffn_drop=0.0, layer_scale_init=1.0):
        super().__init__()
        from einops import rearrange as _r
        self._rearrange = _r
        self.height = height
        self.width  = width
        self.num_spatial = height * width
        self.norm1  = nn.LayerNorm(dim)
        self.norm2  = nn.LayerNorm(dim)
        self.sc_att = SCAtt(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Dropout(ffn_drop),
            nn.Linear(dim * 4, dim),
        )
        self.ls1 = nn.Parameter(torch.ones(dim) * layer_scale_init)
        self.ls2 = nn.Parameter(torch.ones(dim) * layer_scale_init)


def task1_params():
    _heading("TASK 1 — PARAMETER BREAKDOWN")

    cfg   = get_config("aqua20_dual_hybrid_128_focal_balanced")
    model = build_model(cfg, "cpu")

    def count(m):
        return sum(p.numel() for p in m.parameters())

    fine_proj  = count(model.fine_branch.input_projection) + count(model.fine_branch.cls_projection)
    fine_pos   = model.fine_branch.pos_embed.numel()
    fine_block = count(model.fine_branch.blocks)
    fine_norm  = count(model.fine_branch.norm)
    fine_total = count(model.fine_branch)

    coarse_proj  = count(model.coarse_branch.input_projection) + count(model.coarse_branch.cls_projection)
    coarse_pos   = model.coarse_branch.pos_embed.numel()
    coarse_block = count(model.coarse_branch.blocks)
    coarse_norm  = count(model.coarse_branch.norm)
    coarse_total = count(model.coarse_branch)

    head_total = count(model.head)
    grand      = count(model)

    # Per-block breakdown (fine branch, depth=1)
    blk = model.fine_branch.blocks[0]
    blk_mamba = count(blk.mamba)
    blk_norm  = count(blk.norm) + count(blk.norm_ffn)
    blk_cbam  = count(blk.sc_att)
    blk_ffn   = count(blk.ffn)
    blk_ls    = blk.layer_scale_1.numel() + blk.layer_scale_2.numel()
    blk_gate  = 1  # fusion_gate scalar
    blk_total = count(blk)

    print("\n── Fine branch (ViT-B/16, 14×14, depth=1, dim=128) ──")
    print(f"  input_projection + cls_projection : {fine_proj:>10,}")
    print(f"  pos_embed (1+196, 128)            : {fine_pos:>10,}")
    print(f"  MarineMambaBlock ×1               : {fine_block:>10,}")
    print(f"    └─ Mamba                        : {blk_mamba:>10,}")
    print(f"    └─ LayerNorm ×2                 : {blk_norm:>10,}")
    print(f"    └─ CBAM (SCAtt)                 : {blk_cbam:>10,}")
    print(f"    └─ FFN (4× expand)              : {blk_ffn:>10,}")
    print(f"    └─ LayerScale ×2 + fusion gate  : {blk_ls + blk_gate:>10,}")
    print(f"  output norm                       : {fine_norm:>10,}")
    print(f"  Fine branch total                 : {fine_total:>10,}")

    print(f"\n── Coarse branch (ViT-B/32, 7×7, depth=1, dim=128) ──")
    print(f"  input_projection + cls_projection : {coarse_proj:>10,}")
    print(f"  pos_embed (1+49, 128)             : {model.coarse_branch.pos_embed.numel():>10,}")
    print(f"  MarineMambaBlock ×1               : {coarse_block:>10,}")
    print(f"  output norm                       : {coarse_norm:>10,}")
    print(f"  Coarse branch total               : {coarse_total:>10,}")

    print(f"\n── MLP Head (LN(256)→Lin(256→512)→GELU→Drop→Lin(512→20)) ──")
    print(f"  Head total                        : {head_total:>10,}")

    print(f"\n── Grand total                      : {grand:>10,}  ({grand/1e6:.4f} M)")

    # Mamba hyperparams (from model construction in model.py line 118)
    # Mamba(d_model=dim, d_state=32, d_conv=4, expand=2)
    # dt_rank is computed inside mamba_ssm as ceil(d_model/16)
    d_model = cfg["dim"]      # 128
    d_state = 32
    d_conv  = 4
    expand  = 2
    dt_rank = math.ceil(d_model / 16)   # = 8
    d_inner = expand * d_model           # = 256

    print(f"\n── Mamba hyperparams ──")
    print(f"  d_model  = {d_model}")
    print(f"  d_state  = {d_state}")
    print(f"  d_conv   = {d_conv}")
    print(f"  expand   = {expand}")
    print(f"  dt_rank  = {dt_rank}  (= ceil({d_model}/16))")
    print(f"  d_inner  = {d_inner}  (= expand × d_model)")

    # No-Mamba variant (FFN block replaces Mamba block, same outer structure)
    model_nm = build_model(cfg, "cpu")
    for branch in (model_nm.fine_branch, model_nm.coarse_branch):
        branch.blocks = nn.ModuleList([
            _FFNBlock(dim=cfg["dim"],
                      height=blk.height,
                      width=blk.width,
                      ffn_drop=cfg.get("ffn_drop", 0.0),
                      layer_scale_init=cfg.get("layer_scale_init", 1.0))
            for blk in branch.blocks
        ])
    nm_total = count(model_nm)
    print(f"\n── No-Mamba (FFN-only) ablation variant ──")
    print(f"  Trainable params                  : {nm_total:>10,}  ({nm_total/1e6:.4f} M)")
    print(f"  Δ vs full model                   : {nm_total - grand:>+10,}")

    result = {
        "fine_branch":   fine_total,
        "coarse_branch": coarse_total,
        "head":          head_total,
        "grand_total":   grand,
        "grand_total_M": round(grand / 1e6, 4),
        "mamba_hyperparams": {
            "d_model": d_model, "d_state": d_state,
            "d_conv": d_conv, "expand": expand,
            "dt_rank": dt_rank, "d_inner": d_inner,
        },
        "no_mamba_variant": {
            "trainable_params": nm_total,
            "trainable_params_M": round(nm_total / 1e6, 4),
        },
        "fine_branch_detail": {
            "projections": fine_proj,
            "pos_embed": fine_pos,
            "mamba_block": fine_block,
            "mamba_block_detail": {
                "mamba": blk_mamba,
                "layernorm_x2": blk_norm,
                "cbam": blk_cbam,
                "ffn": blk_ffn,
                "layerscale_and_gate": blk_ls + blk_gate,
            },
            "output_norm": fine_norm,
        },
    }
    _dump("task1_params", result)

    print(f"\n| Module | Params |")
    print(f"|--------|--------|")
    print(f"| Fine branch (ViT-B/16, 14×14, depth=1) | {fine_total:,} |")
    print(f"| Coarse branch (ViT-B/32, 7×7, depth=1) | {coarse_total:,} |")
    print(f"| MLP head | {head_total:,} |")
    print(f"| **Grand total** | **{grand:,}** ({grand/1e6:.4f} M) |")
    print(f"| No-Mamba variant | {nm_total:,} ({nm_total/1e6:.4f} M) |")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Macro-F1 from existing checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def _eval_checkpoint(ckpt_path, config_name, device):
    """Returns (top1_pct, macro_f1, labels_true, labels_pred, class_list)."""
    cfg   = get_config(config_name)
    model = build_model(cfg, device)
    sd    = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()

    _, testloader = build_dataloaders(cfg)
    all_true, all_pred = [], []
    with torch.inference_mode():
        for batch in testloader:
            fine_f, coarse_f, fine_cls, coarse_cls, targets = batch
            out = model(fine_f.to(device), coarse_f.to(device),
                        fine_cls.to(device), coarse_cls.to(device))
            all_pred.extend(out.argmax(1).cpu().tolist())
            all_true.extend(targets.tolist())

    top1   = 100.0 * sum(p == t for p, t in zip(all_pred, all_true)) / len(all_true)
    macro  = f1_score(all_true, all_pred, average="macro", zero_division=0) * 100.0
    classes = testloader.dataset.classes
    return top1, macro, all_true, all_pred, classes


def task2_macro_f1():
    _heading("TASK 2 — MACRO-F1 FROM EXISTING CHECKPOINTS")

    datasets = {
        "aqua20": {
            "config":   "aqua20_dual_hybrid_128_focal_balanced",
            "ckpt_tpl": "best_model_aqua20_pyramid_hybrid_128_focal_balanced_seed_{s}.pth",
            "expected_top1_mean": 93.46,
            "expected_top1_std":  0.57,
        },
        "sea23": {
            "config":   "sea23_dual_hybrid_128_focal_balanced",
            "ckpt_tpl": "best_model_sea23_pyramid_hybrid_128_focal_balanced_seed_{s}.pth",
            "expected_top1_mean": 95.90,
            "expected_top1_std":  0.80,
        },
    }

    all_results = {}

    for dset, info in datasets.items():
        print(f"\n── {dset.upper()} ──")
        top1s, f1s = [], []
        for seed in SEEDS_11:
            ckpt = os.path.join(PROJECT_ROOT,
                                info["ckpt_tpl"].format(s=seed))
            if not os.path.exists(ckpt):
                print(f"  MISSING checkpoint: {ckpt}")
                sys.exit(1)
            top1, macro, _, _, _ = _eval_checkpoint(
                ckpt, info["config"], DEVICE)
            top1s.append(top1)
            f1s.append(macro)
            print(f"  seed {seed:2d}  top-1={top1:.2f}%  macro-F1={macro:.2f}%")

        mean_top1 = float(np.mean(top1s))
        std_top1  = float(np.std(top1s))
        mean_f1   = float(np.mean(f1s))
        std_f1    = float(np.std(f1s))

        print(f"\n  Top-1 : {mean_top1:.2f} ± {std_top1:.2f}%")
        print(f"  Macro-F1 : {mean_f1:.2f} ± {std_f1:.2f}%")

        # Sanity check
        exp_m = info["expected_top1_mean"]
        exp_s = info["expected_top1_std"]
        top1_ok = abs(mean_top1 - exp_m) < 0.05 and abs(std_top1 - exp_s) < 0.05
        if not top1_ok:
            print(f"\n  *** SANITY CHECK FAILED ***")
            print(f"  Expected top-1 {exp_m:.2f}±{exp_s:.2f}, got {mean_top1:.2f}±{std_top1:.2f}")
            print(f"  Stopping — do not use these numbers in the paper.")
            sys.exit(1)
        print(f"  Sanity check PASSED (top-1 matches expected {exp_m:.2f}±{exp_s:.2f})")

        all_results[dset] = {
            "per_seed": {str(s): {"top1": round(t, 4), "macro_f1": round(f, 4)}
                         for s, t, f in zip(SEEDS_11, top1s, f1s)},
            "top1_mean": round(mean_top1, 4),
            "top1_std":  round(std_top1,  4),
            "f1_mean":   round(mean_f1,   4),
            "f1_std":    round(std_f1,    4),
        }

    _dump("task2_macro_f1", all_results)

    print("\n| Dataset | Top-1 (11 seeds) | Macro-F1 (11 seeds) |")
    print("|---------|-----------------|---------------------|")
    for dset, r in all_results.items():
        print(f"| {dset} | {r['top1_mean']:.2f} ± {r['top1_std']:.2f}% "
              f"| {r['f1_mean']:.2f} ± {r['f1_std']:.2f}% |")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Frozen CLIP baselines
# ─────────────────────────────────────────────────────────────────────────────

class _CLIPConcatDataset(Dataset):
    """Loads pooled ViT-B/16 + ViT-B/32 features → concat 1024-d."""
    def __init__(self, cls_path, split):
        data = torch.load(cls_path, map_location="cpu")[split]
        self.x      = torch.cat([data["ViT-B-16"], data["ViT-B-32"]], dim=1).float()
        self.labels = data["labels"].long()
        self.classes = data["classes"]
        self.targets = self.labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]


def _build_clip_loaders(cls_path, num_classes, batch_size=128, balanced="sqrt"):
    trainset = _CLIPConcatDataset(cls_path, "train")
    testset  = _CLIPConcatDataset(cls_path, "test")

    if balanced in ("sqrt", "uniform"):
        targets      = torch.as_tensor(trainset.targets, dtype=torch.long)
        class_counts = torch.bincount(targets, minlength=num_classes).float()
        weights      = class_counts.rsqrt() if balanced == "sqrt" else class_counts.reciprocal()
        sampler      = WeightedRandomSampler(weights[targets], num_samples=len(trainset),
                                             replacement=True)
        train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler,
                                  num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    return train_loader, test_loader, trainset.classes


def _train_clip_probe(model, train_loader, test_loader, criterion, device,
                      epochs=60, lr=3e-4, wd=0.05, warmup_epochs=5, patience=15):
    optimizer    = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps  = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        t = step - warmup_steps
        return 0.5 * (1.0 + math.cos(math.pi * t / max(1, total_steps - warmup_steps)))

    scheduler   = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_acc    = 0.0
    best_preds  = None
    best_labels = None
    no_improve  = 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        preds, labels = [], []
        with torch.inference_mode():
            for x, y in test_loader:
                out = model(x.to(device))
                preds.extend(out.argmax(1).cpu().tolist())
                labels.extend(y.tolist())

        acc = 100.0 * sum(p == l for p, l in zip(preds, labels)) / len(labels)
        if acc > best_acc:
            best_acc    = acc
            best_preds  = preds[:]
            best_labels = labels[:]
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_acc, best_preds, best_labels


def _clip_zeroshot(cls_path, class_names, device):
    """CLIP zero-shot using ViT-B/16 text encoder against stored pooled features."""
    try:
        import open_clip
    except ImportError:
        print("  open_clip not installed — skipping zero-shot")
        return None, None, None

    print("  Loading ViT-B/16 text encoder ...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", device=device)
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    # Prompt templates (same ensemble as CLIP paper)
    templates = [
        "a photo of a {}.",
        "a photograph of a {}.",
        "an image of a {}.",
        "a picture of a {}.",
    ]

    def _clean(name):
        return (name.replace("_", " ")
                    .replace("InGroups", " in groups")
                    .replace("fishInGroups", "fish in groups"))

    with torch.inference_mode():
        text_embs = []
        for cls in class_names:
            prompts  = [t.format(_clean(cls)) for t in templates]
            tokens   = tokenizer(prompts).to(device)
            embs     = clip_model.encode_text(tokens)
            embs     = embs / embs.norm(dim=-1, keepdim=True)
            text_embs.append(embs.mean(0))
        text_embs = torch.stack(text_embs)   # [C, 512]
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        data     = torch.load(cls_path, map_location=device)["test"]
        img_feat = data["ViT-B-16"].float().to(device)   # [N, 512]  — B/16 only for zero-shot
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        labels   = data["labels"].tolist()

        sims  = img_feat @ text_embs.T    # [N, C]
        preds = sims.argmax(dim=1).tolist()

    top1    = 100.0 * sum(p == l for p, l in zip(preds, labels)) / len(labels)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0) * 100.0
    return top1, macro_f1, preds


def task3_clip_baselines():
    _heading("TASK 3 — FROZEN CLIP BASELINES (AQUA20, seeds 0/1/2)")

    from marinemamba.paths import CLS_FEATURES_PATH
    NUM_CLASSES  = 20
    CLS_PATH     = CLS_FEATURES_PATH    # data/dual_clip_pooled_features.pt
    INPUT_DIM    = 1024                  # concat ViT-B/16 + ViT-B/32

    if not os.path.exists(CLS_PATH):
        print(f"MISSING: {CLS_PATH}")
        sys.exit(1)

    # Build loaders (same for all variants — sqrt sampler, focal alpha on train)
    train_loader, test_loader, class_names = _build_clip_loaders(
        CLS_PATH, NUM_CLASSES, batch_size=128, balanced="sqrt")

    # Pre-compute class counts for focal loss alpha
    train_data   = torch.load(CLS_PATH, map_location="cpu")["train"]
    train_labels = train_data["labels"].long()
    class_counts = torch.bincount(train_labels, minlength=NUM_CLASSES).float().to(DEVICE)

    def _focal_criterion():
        counts = class_counts
        alpha  = (counts.sum() / (NUM_CLASSES * counts)).clamp(min=1e-6)
        alpha  = alpha / alpha.sum() * NUM_CLASSES
        return FocalLoss(gamma=2.0, alpha=alpha).to(DEVICE)

    def _ce_criterion():
        return nn.CrossEntropyLoss()

    def _count(m):
        return sum(p.numel() for p in m.parameters())

    probe_defs = {
        "a_linear_ce": {
            "label": "Linear(1024, C) — CE",
            "model_fn": lambda: nn.Linear(INPUT_DIM, NUM_CLASSES),
            "loss_fn":  _ce_criterion,
        },
        "b_mlp_ce": {
            "label": "LN→Lin(1024,256)→GELU→Drop(0.3)→Lin(256,C) — CE",
            "model_fn": lambda: nn.Sequential(
                nn.LayerNorm(INPUT_DIM),
                nn.Linear(INPUT_DIM, 256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, NUM_CLASSES),
            ),
            "loss_fn": _ce_criterion,
        },
        "c_mlp_focal": {
            "label": "LN→Lin(1024,256)→GELU→Drop(0.3)→Lin(256,C) — Focal",
            "model_fn": lambda: nn.Sequential(
                nn.LayerNorm(INPUT_DIM),
                nn.Linear(INPUT_DIM, 256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, NUM_CLASSES),
            ),
            "loss_fn": _focal_criterion,
        },
    }

    all_results = {}
    rows = []

    for key, pdef in probe_defs.items():
        print(f"\n── {pdef['label']} ──")
        probe_model = pdef["model_fn"]()
        n_params = _count(probe_model)
        print(f"  Params: {n_params:,}")

        seed_top1s, seed_f1s = [], []
        for seed in SEEDS_3:
            set_seed(seed)
            model = pdef["model_fn"]().to(DEVICE)
            crit  = pdef["loss_fn"]()
            acc, preds, labels = _train_clip_probe(
                model, train_loader, test_loader, crit, DEVICE)
            macro_f1 = f1_score(labels, preds, average="macro", zero_division=0) * 100.0
            seed_top1s.append(acc)
            seed_f1s.append(macro_f1)
            print(f"  seed {seed}  top-1={acc:.2f}%  macro-F1={macro_f1:.2f}%")

        mean_top1 = float(np.mean(seed_top1s))
        std_top1  = float(np.std(seed_top1s))
        mean_f1   = float(np.mean(seed_f1s))
        std_f1    = float(np.std(seed_f1s))

        print(f"  → Top-1: {mean_top1:.2f} ± {std_top1:.2f}%  "
              f"Macro-F1: {mean_f1:.2f} ± {std_f1:.2f}%")

        all_results[key] = {
            "label":    pdef["label"],
            "params":   n_params,
            "per_seed": {str(s): {"top1": round(t, 4), "macro_f1": round(f, 4)}
                         for s, t, f in zip(SEEDS_3, seed_top1s, seed_f1s)},
            "top1_mean": round(mean_top1, 4),
            "top1_std":  round(std_top1,  4),
            "f1_mean":   round(mean_f1,   4),
            "f1_std":    round(std_f1,    4),
        }
        rows.append((pdef["label"], n_params, mean_top1, std_top1, mean_f1, std_f1))

    # Zero-shot
    print("\n── CLIP zero-shot (ViT-B/16, class-name prompts) ──")
    top1_zs, f1_zs, preds_zs = _clip_zeroshot(CLS_PATH, class_names, DEVICE)
    if top1_zs is not None:
        print(f"  Top-1: {top1_zs:.2f}%  Macro-F1: {f1_zs:.2f}%")
        all_results["d_zeroshot"] = {
            "label":   "CLIP zero-shot (ViT-B/16, 4-template ensemble)",
            "params":  0,
            "top1":    round(top1_zs, 4),
            "macro_f1": round(f1_zs, 4),
        }
        rows.append(("CLIP zero-shot (ViT-B/16)", 0, top1_zs, 0.0, f1_zs, 0.0))

    _dump("task3_clip_baselines", all_results)

    # Ablation reference rows (from existing results)
    print("\n── Comparison against paper's ablation ──")
    ablation_ref = {
        "Full model (MarineMamba)":        (93.44, 0.13),
        "No Mamba (FFN only)":             (85.98, 0.44),
        "No CLIP Feature Vector injection": (85.07, 0.61),
    }

    print("\n| Method | Params | Top-1 (AQUA20) | Macro-F1 |")
    print("|--------|--------|----------------|----------|")
    for label, n_params, top1_m, top1_s, f1_m, f1_s in rows:
        top1_str = f"{top1_m:.2f} ± {top1_s:.2f}%" if top1_s > 0 else f"{top1_m:.2f}%"
        f1_str   = f"{f1_m:.2f} ± {f1_s:.2f}%"     if f1_s > 0 else f"{f1_m:.2f}%"
        print(f"| {label} | {n_params:,} | {top1_str} | {f1_str} |")

    print(f"\n  Ablation reference (3 seeds each, from paper):")
    for name, (m, s) in ablation_ref.items():
        print(f"  {name:<42}  {m:.2f} ± {s:.2f}%")

    # Explicit warning if any baseline beats No-Mamba
    nm_threshold = 85.98
    for key, r in all_results.items():
        top1 = r.get("top1_mean") or r.get("top1")
        if top1 is not None and top1 > nm_threshold:
            print(f"\n  *** INTERPRETATION NOTE ***")
            print(f"  {r['label']} achieves {top1:.2f}% top-1,")
            print(f"  which is ABOVE No-Mamba (FFN only) = {nm_threshold:.2f}%.")
            print(f"  This means the −{93.44 - nm_threshold:.2f}pp attributed to removing Mamba")
            print(f"  is measuring a weak spatial variant vs frozen-CLIP concat,")
            print(f"  not the marginal value of the state-space layer alone.")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Fish4K per-class analysis
# ─────────────────────────────────────────────────────────────────────────────

def task4_fish4k_perclass():
    _heading("TASK 4 — FISH4KNOWLEDGE PER-CLASS ANALYSIS")

    # ── VIM — MISSING CHECKPOINTS ─────────────────────────────────────────────
    print("""
  *** Vim-tiny checkpoints: NOT FOUND ***
  scripts/baseline_vim.py never calls torch.save — only best accuracy is logged
  to results_vim_fish4k.json and logs/log_vim_fish4k_seed_N.txt.
  Per-class recall/F1 and confusion matrices for Vim cannot be computed without
  rerunning training with checkpoint saving.
  The Vim comparison in Task 4 is incomplete; only MarineMamba is reported.
""")

    # ── MARINMAMBA on fish4k ──────────────────────────────────────────────────
    cfg = get_config("fish4k_dual_hybrid_128_focal_balanced")
    _, testloader = build_dataloaders(cfg)
    class_names = testloader.dataset.classes    # ['fish_01', ..., 'fish_23']
    num_classes  = cfg["num_classes"]           # 23

    # Train class distribution (from pooled features, same labels)
    from marinemamba.paths import FISH4K_CLS_PATH
    fish_cls = torch.load(FISH4K_CLS_PATH, map_location="cpu")
    train_labels_all = fish_cls["train"]["labels"]
    train_counts = torch.bincount(train_labels_all, minlength=num_classes).tolist()
    test_labels_all  = fish_cls["test"]["labels"]
    test_counts  = torch.bincount(test_labels_all,  minlength=num_classes).tolist()

    max_cls = int(np.argmax(train_counts))
    min_cls = int(np.argmin(train_counts))
    print(f"  Largest  class : {class_names[max_cls]} — {train_counts[max_cls]:,} train / {test_counts[max_cls]:,} test")
    print(f"  Smallest class : {class_names[min_cls]} — {train_counts[min_cls]:,} train / {test_counts[min_cls]:,} test")
    print(f"  Imbalance ratio: {train_counts[max_cls] / train_counts[min_cls]:.0f}:1  "
          f"({train_counts[max_cls]:,} / {train_counts[min_cls]:,})")

    # 5 most frequent, 10 rarest
    sorted_by_freq = sorted(range(num_classes), key=lambda i: train_counts[i], reverse=True)
    top5_idx    = sorted_by_freq[:5]
    bottom10_idx = sorted_by_freq[-10:]

    CKPT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
    all_cms    = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_class_recalls_seeds = np.zeros((len(CKPT_SEEDS), num_classes))
    per_class_f1s_seeds     = np.zeros((len(CKPT_SEEDS), num_classes))
    seed_macro_f1s = []
    seed_top1s     = []

    for si, seed in enumerate(CKPT_SEEDS):
        ckpt_path = os.path.join(PROJECT_ROOT,
                                 f"best_model_fish4k_baseline_seed_{seed}.pth")
        if not os.path.exists(ckpt_path):
            print(f"  MISSING checkpoint: {ckpt_path}")
            sys.exit(1)

        model = build_model(cfg, DEVICE)
        sd    = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(sd, strict=True)
        model.eval()

        all_true, all_pred = [], []
        with torch.inference_mode():
            for batch in testloader:
                ff, cf, fc, cc, tgt = batch
                out = model(ff.to(DEVICE), cf.to(DEVICE), fc.to(DEVICE), cc.to(DEVICE))
                all_pred.extend(out.argmax(1).cpu().tolist())
                all_true.extend(tgt.tolist())

        top1 = 100.0 * sum(p == t for p, t in zip(all_pred, all_true)) / len(all_true)
        macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0) * 100.0
        seed_top1s.append(top1)
        seed_macro_f1s.append(macro_f1)

        cm = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))
        all_cms += cm

        # Per-class recall and F1
        per_f1     = f1_score(all_true, all_pred, average=None,
                               labels=list(range(num_classes)), zero_division=0)
        per_recall = cm.diagonal() / cm.sum(axis=1).clip(min=1)
        per_class_recalls_seeds[si] = per_recall
        per_class_f1s_seeds[si]     = per_f1

        print(f"  seed {seed:2d}  top-1={top1:.2f}%  macro-F1={macro_f1:.2f}%")

        # Save per-seed confusion matrix
        npy_path = os.path.join(RESULTS_DIR, f"cm_fish4k_marinemamba_seed_{seed}.npy")
        np.save(npy_path, cm)

    # Aggregate confusion matrix
    agg_cm_path = os.path.join(RESULTS_DIR, "cm_fish4k_marinemamba_agg.npy")
    np.save(agg_cm_path, all_cms)
    print(f"\n  Confusion matrices saved to results/")

    mean_recall = per_class_recalls_seeds.mean(axis=0)
    std_recall  = per_class_recalls_seeds.std(axis=0)
    mean_f1     = per_class_f1s_seeds.mean(axis=0)
    std_f1      = per_class_f1s_seeds.std(axis=0)

    overall_top1_mean  = float(np.mean(seed_top1s))
    overall_top1_std   = float(np.std(seed_top1s))
    overall_macro_mean = float(np.mean(seed_macro_f1s))
    overall_macro_std  = float(np.std(seed_macro_f1s))

    print(f"\n  MarineMamba fish4k (11 seeds)")
    print(f"  Top-1    : {overall_top1_mean:.2f} ± {overall_top1_std:.2f}%")
    print(f"  Macro-F1 : {overall_macro_mean:.2f} ± {overall_macro_std:.2f}%")

    # Recall over 5 most frequent vs 10 rarest (mean over seeds)
    recall_top5    = mean_recall[top5_idx].mean() * 100.0
    recall_bot10   = mean_recall[bottom10_idx].mean() * 100.0
    print(f"\n  Recall (5 most frequent classes)  : {recall_top5:.2f}%")
    print(f"  Recall (10 rarest classes)        : {recall_bot10:.2f}%")

    # Per-class table sorted by train frequency (descending)
    print(f"\n| Rank | Class | Train | Test | Recall (mean±std) | F1 (mean±std) |")
    print(f"|------|-------|-------|------|-------------------|---------------|")
    for rank, cidx in enumerate(sorted_by_freq, 1):
        print(f"| {rank:2d} | {class_names[cidx]} | {train_counts[cidx]:5,} | "
              f"{test_counts[cidx]:4,} | "
              f"{mean_recall[cidx]*100:.1f} ± {std_recall[cidx]*100:.1f}% | "
              f"{mean_f1[cidx]*100:.1f} ± {std_f1[cidx]*100:.1f}% |")

    # Test hypothesis: rare-class recall > common-class recall?
    print(f"\n  Hypothesis (focal+sqrt sampler → better rare-class recall):")
    print(f"  5 most frequent mean recall : {recall_top5:.2f}%")
    print(f"  10 rarest mean recall       : {recall_bot10:.2f}%")
    if recall_bot10 > recall_top5:
        print(f"  ✓ SUPPORTED — rare classes have HIGHER recall than common classes.")
        print(f"  Gap: +{recall_bot10 - recall_top5:.2f}pp for rare classes.")
    else:
        print(f"  ✗ NOT SUPPORTED — common classes still have higher recall.")
        print(f"  Gap: {recall_top5 - recall_bot10:.2f}pp for common > rare.")

    result = {
        "vim_status": "MISSING — baseline_vim.py never saves checkpoints; "
                      "Vim per-class metrics require rerunning with torch.save",
        "marinemamba": {
            "top1_mean":   round(overall_top1_mean, 4),
            "top1_std":    round(overall_top1_std, 4),
            "macro_f1_mean": round(overall_macro_mean, 4),
            "macro_f1_std":  round(overall_macro_std, 4),
            "imbalance_ratio": f"{train_counts[max_cls]}:{train_counts[min_cls]}",
            "largest_class":  {"name": class_names[max_cls], "train": train_counts[max_cls], "test": test_counts[max_cls]},
            "smallest_class": {"name": class_names[min_cls], "train": train_counts[min_cls], "test": test_counts[min_cls]},
            "recall_top5_freq":  round(recall_top5, 4),
            "recall_bot10_rare": round(recall_bot10, 4),
            "per_class": {
                class_names[cidx]: {
                    "train_count":  train_counts[cidx],
                    "test_count":   test_counts[cidx],
                    "recall_mean":  round(float(mean_recall[cidx]) * 100, 2),
                    "recall_std":   round(float(std_recall[cidx])  * 100, 2),
                    "f1_mean":      round(float(mean_f1[cidx])     * 100, 2),
                    "f1_std":       round(float(std_f1[cidx])      * 100, 2),
                }
                for cidx in sorted_by_freq
            },
            "confusion_matrices": {
                "per_seed": [f"results/cm_fish4k_marinemamba_seed_{s}.npy" for s in CKPT_SEEDS],
                "aggregated": "results/cm_fish4k_marinemamba_agg.npy",
            },
        },
    }
    _dump("task4_fish4k_perclass", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    r1 = task1_params()
    r2 = task2_macro_f1()
    r3 = task3_clip_baselines()
    r4 = task4_fish4k_perclass()

    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETE — results saved to results/")
    print("=" * 70)
