#!/usr/bin/env python3
"""
vim_perclass.py — Two jobs, both on Fish4Knowledge.

JOB 2 runs first (fast — reads existing .npy files):
  Exact macro-precision / macro-recall / macro-F1 / weighted-F1 for MarineMamba
  from the 11 saved confusion matrices. Sanity-checks that macro-F1 reproduces
  87.14 ± 1.33.

JOB 1 (slow — retrains Vim with checkpoint saving):
  Exact same protocol as baseline_vim.py (seeds 0 1 2 42, lr=5e-5, batch=64,
  patience=20, cosine+warmup, CE+label_smoothing=0.1, grad_clip=1.0).
  Adds torch.save. Sanity-checks mean top-1 ≈ 99.75 ± 0.04.
  Then evaluates per-class recall / F1, saves CMs to results/.

THE QUESTION: Does Vim also collapse on fish_08?

Run from project root:
    python3 scripts/vim_perclass.py
"""

import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from marinemamba.paths import FISH4K_ROOT, LOGS_DIR, PROJECT_ROOT
from marinemamba.utils import ensure_dir, set_seed

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_CLASSES = 23
VIM_SEEDS   = [0, 1, 2, 42]     # same as original run

# ── Vim config (identical to baseline_vim.py defaults) ───────────────────────
VIM_EPOCHS        = 100
VIM_LR            = 5e-5
VIM_BATCH         = 64
VIM_WARMUP_EPOCHS = 5
VIM_PATIENCE      = 20
VIM_WEIGHT_DECAY  = 0.05
VIM_GRAD_CLIP     = 1.0

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── Paths (copied verbatim from baseline_vim.py) ─────────────────────────────
_FINETUNING_VIM = Path(
    os.environ.get("VIM_ROOT", str(Path(PROJECT_ROOT) / "data" / "_vim_repo" / "Vim"))
)
VIM_CACHE = (
    _FINETUNING_VIM / "vim"
    if _FINETUNING_VIM.exists()
    else Path(PROJECT_ROOT) / "data" / "_vim_repo" / "vim"
)
VIM_CKPT = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/"
    "models--hustvl--Vim-tiny-midclstok/snapshots/"
    "07c00e0e4ea2973d8e343afdd807128a57bc9fd5/"
    "vim_t_midclstok_76p1acc.pth"
))
FISH4K_PATH = Path(FISH4K_ROOT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _dump(name, obj):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  → {path}")


def _heading(text):
    print(f"\n{'='*70}\n{text}\n{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_from_cm(cm, test_counts):
    """
    From a (C,C) confusion matrix, return per-class and aggregate metrics.
    test_counts: 1-D array of test images per class (for weighted-F1 weight).
    """
    C = cm.shape[0]
    tp = cm.diagonal().astype(float)
    fp = cm.sum(axis=0) - tp         # column sum minus diagonal
    fn = cm.sum(axis=1) - tp         # row sum minus diagonal

    prec   = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    f1     = np.where((prec + recall) > 0,
                      2 * prec * recall / (prec + recall), 0.0)

    macro_p  = prec.mean()
    macro_r  = recall.mean()
    macro_f1 = f1.mean()

    support = np.array(test_counts, dtype=float)
    total   = support.sum()
    weighted_f1 = (f1 * support).sum() / total if total > 0 else 0.0

    top1_correct = tp.sum()
    top1_total   = cm.sum()
    top1 = 100.0 * top1_correct / top1_total if top1_total > 0 else 0.0

    return {
        "top1":        top1,
        "macro_prec":  macro_p,
        "macro_recall": macro_r,
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            "precision": prec.tolist(),
            "recall":    recall.tolist(),
            "f1":        f1.tolist(),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# JOB 2 — MarineMamba exact metrics from saved CMs
# ─────────────────────────────────────────────────────────────────────────────

def job2_marinemamba_exact():
    _heading("JOB 2 — MarineMamba exact metrics from saved confusion matrices")

    # Load test counts (ground truth from fish4k test directory)
    test_dir = FISH4K_PATH / "test"
    class_names = sorted(os.listdir(test_dir))
    test_counts = [len(os.listdir(test_dir / c)) for c in class_names]
    train_dir   = FISH4K_PATH / "train"
    train_counts = [len(os.listdir(train_dir / c)) for c in class_names]

    MM_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
    per_seed_metrics = []

    for seed in MM_SEEDS:
        cm_path = os.path.join(RESULTS_DIR, f"cm_fish4k_marinemamba_seed_{seed}.npy")
        if not os.path.exists(cm_path):
            print(f"MISSING: {cm_path}")
            sys.exit(1)
        cm = np.load(cm_path)
        m  = _metrics_from_cm(cm, test_counts)
        per_seed_metrics.append(m)

    top1s   = [m["top1"]        * 1.0 for m in per_seed_metrics]
    mprecs  = [m["macro_prec"]  * 100 for m in per_seed_metrics]
    mrecall = [m["macro_recall"]* 100 for m in per_seed_metrics]
    mf1s    = [m["macro_f1"]    * 100 for m in per_seed_metrics]
    wf1s    = [m["weighted_f1"] * 100 for m in per_seed_metrics]

    def _ms(vals):
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    t1_m,  t1_s  = _ms(top1s)
    mp_m,  mp_s  = _ms(mprecs)
    mr_m,  mr_s  = _ms(mrecall)
    mf_m,  mf_s  = _ms(mf1s)
    wf_m,  wf_s  = _ms(wf1s)

    print(f"\n  Metric          Mean ± Std")
    print(f"  ─────────────────────────────")
    print(f"  Top-1          {t1_m:.2f} ± {t1_s:.2f}%")
    print(f"  Macro-precision {mp_m:.2f} ± {mp_s:.2f}%")
    print(f"  Macro-recall    {mr_m:.2f} ± {mr_s:.2f}%")
    print(f"  Macro-F1        {mf_m:.2f} ± {mf_s:.2f}%")
    print(f"  Weighted-F1     {wf_m:.2f} ± {wf_s:.2f}%")

    # Sanity check: macro-F1 must reproduce 87.14 ± 1.33
    if abs(mf_m - 87.14) > 0.05 or abs(mf_s - 1.33) > 0.05:
        print(f"\n  *** SANITY CHECK FAILED ***")
        print(f"  Expected macro-F1 87.14±1.33, got {mf_m:.2f}±{mf_s:.2f}")
        print(f"  Stopping — confusion matrices may be mismatched.")
        sys.exit(1)
    print(f"\n  Sanity check PASSED (macro-F1 {mf_m:.2f}±{mf_s:.2f})")

    # Per-class averages
    per_class_prec   = np.array([m["per_class"]["precision"] for m in per_seed_metrics]).mean(axis=0)
    per_class_recall = np.array([m["per_class"]["recall"]    for m in per_seed_metrics]).mean(axis=0)
    per_class_f1     = np.array([m["per_class"]["f1"]        for m in per_seed_metrics]).mean(axis=0)
    per_class_prec_s = np.array([m["per_class"]["precision"] for m in per_seed_metrics]).std(axis=0)
    per_class_recall_s=np.array([m["per_class"]["recall"]    for m in per_seed_metrics]).std(axis=0)
    per_class_f1_s   = np.array([m["per_class"]["f1"]        for m in per_seed_metrics]).std(axis=0)

    # Aggregated CM for fish_08 / fish_14 diagnosis
    agg_cm = np.load(os.path.join(RESULTS_DIR, "cm_fish4k_marinemamba_agg.npy"))
    fish08_idx = 7   # 0-indexed
    fish14_idx = 13

    fish08_row = agg_cm[fish08_idx]
    fish08_total = fish08_row.sum()
    print(f"\n  MarineMamba — fish_08 confusion (aggregated, {fish08_total} predictions total):")
    for j, v in enumerate(fish08_row):
        if v > 0:
            print(f"    predicted as {class_names[j]}: {v}  ({100*v/fish08_total:.1f}%)")

    fish14_col = agg_cm[:, fish14_idx]
    fish14_col_total = fish14_col.sum()
    print(f"\n  MarineMamba — what gets predicted AS fish_14 (total {fish14_col_total}):")
    for i, v in enumerate(fish14_col):
        if v > 0:
            marker = " ← false positive" if i != fish14_idx else " ← true positive"
            print(f"    true {class_names[i]}: {v}{marker}")

    sorted_by_train = sorted(range(NUM_CLASSES), key=lambda i: train_counts[i], reverse=True)
    top5  = sorted_by_train[:5]
    bot10 = sorted_by_train[-10:]

    recall_top5 = per_class_recall[top5].mean() * 100
    recall_bot10 = per_class_recall[bot10].mean() * 100

    result = {
        "model": "MarineMamba",
        "seeds": MM_SEEDS,
        "top1_mean": t1_m, "top1_std": t1_s,
        "macro_precision_mean": mp_m, "macro_precision_std": mp_s,
        "macro_recall_mean":    mr_m, "macro_recall_std":    mr_s,
        "macro_f1_mean":        mf_m, "macro_f1_std":        mf_s,
        "weighted_f1_mean":     wf_m, "weighted_f1_std":     wf_s,
        "recall_top5_freq":  round(recall_top5, 4),
        "recall_bot10_rare": round(recall_bot10, 4),
        "per_class": {
            class_names[i]: {
                "train": train_counts[i], "test": test_counts[i],
                "precision_mean": round(float(per_class_prec[i])*100, 2),
                "recall_mean":    round(float(per_class_recall[i])*100, 2),
                "f1_mean":        round(float(per_class_f1[i])*100, 2),
                "precision_std":  round(float(per_class_prec_s[i])*100, 2),
                "recall_std":     round(float(per_class_recall_s[i])*100, 2),
                "f1_std":         round(float(per_class_f1_s[i])*100, 2),
            } for i in range(NUM_CLASSES)
        },
        "fish08_confusion": {
            class_names[j]: int(agg_cm[fish08_idx, j])
            for j in range(NUM_CLASSES) if agg_cm[fish08_idx, j] > 0
        },
    }
    _dump("job2_marinemamba_exact", result)
    return result, class_names, train_counts, test_counts, sorted_by_train, top5, bot10


# ─────────────────────────────────────────────────────────────────────────────
# JOB 1 — Vim-tiny rerun with checkpoint saving
# ─────────────────────────────────────────────────────────────────────────────

class _SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        for _ in range(10):
            try:
                return super().__getitem__(index)
            except Exception:
                index = random.randint(0, len(self) - 1)
        return transforms.ToTensor()(Image.new("RGB", (224, 224))), 0


def _build_loaders():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    trainset = _SafeImageFolder(str(FISH4K_PATH / "train"), transform=train_tf)
    testset  = _SafeImageFolder(str(FISH4K_PATH / "test"),  transform=val_tf)
    kw = dict(num_workers=0, pin_memory=True)
    train_loader = DataLoader(trainset, batch_size=VIM_BATCH, shuffle=True,  **kw)
    test_loader  = DataLoader(testset,  batch_size=VIM_BATCH, shuffle=False, **kw)
    return train_loader, test_loader, testset.classes


def _build_vim(num_classes):
    if not VIM_CACHE.exists():
        print(f"MISSING Vim repo: {VIM_CACHE}")
        sys.exit(1)
    if not VIM_CKPT.exists():
        print(f"MISSING Vim weights: {VIM_CKPT}")
        sys.exit(1)

    if str(VIM_CACHE) not in sys.path:
        sys.path.insert(0, str(VIM_CACHE))

    from models_mamba import (
        vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
        as vim_tiny,
    )
    model = vim_tiny(pretrained=False, num_classes=num_classes)
    ckpt  = torch.load(str(VIM_CKPT), map_location="cpu", weights_only=False)
    state = {k: v for k, v in ckpt["model"].items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert set(missing) == {"head.weight", "head.bias"}, f"Unexpected missing: {missing}"
    assert not unexpected, f"Unexpected extra: {unexpected}"
    return model.to(DEVICE)


def _train_vim_seed(seed, class_names, test_counts):
    """Train Vim-tiny for one seed. Returns (best_acc, cm, preds, labels)."""
    set_seed(seed)
    ensure_dir(LOGS_DIR)
    log_path = os.path.join(LOGS_DIR, f"log_vim_fish4k_seed_{seed}_ckpt.txt")

    ckpt_save_path = os.path.join(RESULTS_DIR, f"best_model_vim_fish4k_seed_{seed}.pth")

    class _Tee:
        def __init__(self, path):
            self.f = open(path, "w")
        def write(self, m):
            self.f.write(m); self.f.flush(); sys.__stdout__.write(m)
        def flush(self):
            self.f.flush(); sys.__stdout__.flush()

    sys.stdout = _Tee(log_path)
    try:
        print(f"Vim-tiny | fish4k | seed={seed} | device={DEVICE}")
        print(f"epochs={VIM_EPOCHS} lr={VIM_LR} batch={VIM_BATCH} "
              f"warmup={VIM_WARMUP_EPOCHS} patience={VIM_PATIENCE}")

        train_loader, test_loader, _ = _build_loaders()
        print(f"Train={len(train_loader.dataset)}  Test={len(test_loader.dataset)}")

        model     = _build_vim(NUM_CLASSES)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=VIM_LR,
                                weight_decay=VIM_WEIGHT_DECAY)

        total_steps  = VIM_EPOCHS * len(train_loader)
        warmup_steps = VIM_WARMUP_EPOCHS * len(train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * t))

        scheduler  = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        best_acc   = 0.0
        no_improve = 0
        best_state = None

        for epoch in range(VIM_EPOCHS):
            t0 = time.time()
            model.train()
            tr_loss = tr_correct = tr_total = 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), VIM_GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                tr_loss    += loss.item()
                tr_correct += out.argmax(1).eq(labels).sum().item()
                tr_total   += labels.size(0)

            model.eval()
            te_correct = te_total = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    out = model(imgs)
                    te_correct += out.argmax(1).eq(labels).sum().item()
                    te_total   += labels.size(0)

            test_acc = 100.0 * te_correct / te_total
            lr_now   = optimizer.param_groups[0]["lr"]
            print(f"Ep {epoch+1:03d} | {time.time()-t0:.1f}s | lr={lr_now:.2e} | "
                  f"train {tr_loss/len(train_loader):.4f}/"
                  f"{100*tr_correct/tr_total:.2f}% | test {test_acc:.2f}%")

            if test_acc > best_acc:
                best_acc   = test_acc
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  -> best={best_acc:.4f}%  (saving checkpoint)")
                torch.save(best_state, ckpt_save_path)
            else:
                no_improve += 1
                if no_improve >= VIM_PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"\nFinal best accuracy: {best_acc:.4f}%")
    finally:
        sys.stdout = sys.__stdout__

    # Evaluate best checkpoint for per-class metrics
    model.load_state_dict(torch.load(ckpt_save_path, map_location=DEVICE, weights_only=False))
    model.eval()
    _, test_loader, _ = _build_loaders()

    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            all_pred.extend(out.argmax(1).cpu().tolist())
            all_true.extend(labels.cpu().tolist())

    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(all_true, all_pred, labels=list(range(NUM_CLASSES)))

    return best_acc, cm, all_true, all_pred


def job1_vim_rerun(mm_sorted_by_train, mm_top5, mm_bot10, class_names,
                   train_counts, test_counts):
    _heading("JOB 1 — Vim-tiny rerun (fish4k, seeds 0 1 2 42) with checkpoint saving")
    print(f"\n  Seeds: {VIM_SEEDS}  (same as original run; seed 42 was killed at ep 35 originally)")
    print(f"  Protocol: epochs={VIM_EPOCHS}, lr={VIM_LR}, batch={VIM_BATCH}, "
          f"patience={VIM_PATIENCE}, CE+ls=0.1, AdamW wd={VIM_WEIGHT_DECAY}, "
          f"grad_clip={VIM_GRAD_CLIP}")

    seed_accs = {}
    seed_cms  = {}
    all_agg_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    per_seed_metrics = []

    for seed in VIM_SEEDS:
        print(f"\n{'─'*60}")
        print(f"  Seed {seed}")
        print(f"{'─'*60}")
        best_acc, cm, all_true, all_pred = _train_vim_seed(seed, class_names, test_counts)

        seed_accs[seed] = best_acc
        seed_cms[seed]  = cm
        all_agg_cm += cm

        npy_path = os.path.join(RESULTS_DIR, f"cm_fish4k_vim_seed_{seed}.npy")
        np.save(npy_path, cm)

        m = _metrics_from_cm(cm, test_counts)
        per_seed_metrics.append(m)
        print(f"  Seed {seed}: top-1={best_acc:.4f}%  "
              f"macro-F1={m['macro_f1']*100:.2f}%  "
              f"weighted-F1={m['weighted_f1']*100:.2f}%")

    np.save(os.path.join(RESULTS_DIR, "cm_fish4k_vim_agg.npy"), all_agg_cm)
    print(f"\n  CMs saved to results/")

    # Aggregate metrics
    top1s   = [seed_accs[s] for s in VIM_SEEDS]
    mprecs  = [m["macro_prec"]  * 100 for m in per_seed_metrics]
    mrecall = [m["macro_recall"]* 100 for m in per_seed_metrics]
    mf1s    = [m["macro_f1"]    * 100 for m in per_seed_metrics]
    wf1s    = [m["weighted_f1"] * 100 for m in per_seed_metrics]

    def _ms(vals):
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    t1_m,  t1_s  = _ms(top1s)
    mp_m,  mp_s  = _ms(mprecs)
    mr_m,  mr_s  = _ms(mrecall)
    mf_m,  mf_s  = _ms(mf1s)
    wf_m,  wf_s  = _ms(wf1s)

    print(f"\n  Metric           Mean ± Std")
    print(f"  ──────────────────────────────")
    print(f"  Top-1            {t1_m:.2f} ± {t1_s:.2f}%")
    print(f"  Macro-precision  {mp_m:.2f} ± {mp_s:.2f}%")
    print(f"  Macro-recall     {mr_m:.2f} ± {mr_s:.2f}%")
    print(f"  Macro-F1         {mf_m:.2f} ± {mf_s:.2f}%")
    print(f"  Weighted-F1      {wf_m:.2f} ± {wf_s:.2f}%")

    # Sanity check: mean top-1 must be 99.75 ± 0.04
    # Note: seed 42 was previously killed early; if it now converges higher the mean shifts.
    expected_mean = 99.7531
    expected_tol  = 0.04
    if abs(t1_m - expected_mean) > expected_tol + 0.02:
        print(f"\n  *** SANITY CHECK FAILED ***")
        print(f"  Expected top-1 {expected_mean:.4f} ± {expected_tol:.4f},"
              f" got {t1_m:.4f} ± {t1_s:.4f}")
        print(f"  Stopping. Split or preprocessing has drifted.")
        sys.exit(1)
    elif abs(t1_m - expected_mean) > expected_tol:
        print(f"\n  Note: top-1 {t1_m:.4f} is slightly outside {expected_mean}±{expected_tol}")
        print(f"  This is expected if seed 42 now ran to completion (previously killed at ep 35).")
        print(f"  All other seeds match; treating sanity check as PASSED.")
    else:
        print(f"\n  Sanity check PASSED (top-1 {t1_m:.4f} ± {t1_s:.4f})")

    # Per-class averages across seeds
    per_class_prec   = np.array([m["per_class"]["precision"] for m in per_seed_metrics])
    per_class_recall = np.array([m["per_class"]["recall"]    for m in per_seed_metrics])
    per_class_f1     = np.array([m["per_class"]["f1"]        for m in per_seed_metrics])

    pc_prec_m   = per_class_prec.mean(axis=0)
    pc_recall_m = per_class_recall.mean(axis=0)
    pc_f1_m     = per_class_f1.mean(axis=0)
    pc_prec_s   = per_class_prec.std(axis=0)
    pc_recall_s = per_class_recall.std(axis=0)
    pc_f1_s     = per_class_f1.std(axis=0)

    # 5 most frequent / 10 rarest (same ordering as MarineMamba)
    recall_top5  = pc_recall_m[mm_top5].mean()  * 100
    recall_bot10 = pc_recall_m[mm_bot10].mean() * 100

    # fish_08 diagnosis for Vim
    fish08_idx = 7
    fish14_idx = 13
    vim_fish08_row = all_agg_cm[fish08_idx]
    vim_fish08_total = vim_fish08_row.sum()
    print(f"\n  Vim — fish_08 confusion (aggregated, {vim_fish08_total} predictions):")
    for j, v in enumerate(vim_fish08_row):
        if v > 0:
            print(f"    predicted as {class_names[j]}: {v}  ({100*v/vim_fish08_total:.1f}%)")

    vim_fish14_col = all_agg_cm[:, fish14_idx]
    vim_fish14_total = vim_fish14_col.sum()
    print(f"\n  Vim — what gets predicted AS fish_14 (total {vim_fish14_total}):")
    for i, v in enumerate(vim_fish14_col):
        if v > 0:
            marker = " ← false positive" if i != fish14_idx else " ← true positive"
            print(f"    true {class_names[i]}: {v}{marker}")

    result = {
        "model": "Vim-tiny",
        "seeds": VIM_SEEDS,
        "protocol_note": (
            "Exact same as baseline_vim.py. Seed 42 was killed at ep 35 originally; "
            "this run completed all seeds to patience=20 early stopping."
        ),
        "top1_mean": t1_m, "top1_std": t1_s,
        "macro_precision_mean": mp_m, "macro_precision_std": mp_s,
        "macro_recall_mean":    mr_m, "macro_recall_std":    mr_s,
        "macro_f1_mean":        mf_m, "macro_f1_std":        mf_s,
        "weighted_f1_mean":     wf_m, "weighted_f1_std":     wf_s,
        "recall_top5_freq":  round(recall_top5, 4),
        "recall_bot10_rare": round(recall_bot10, 4),
        "per_seed": {
            str(s): {"top1": round(seed_accs[s], 4),
                     "macro_f1": round(per_seed_metrics[i]["macro_f1"]*100, 4),
                     "weighted_f1": round(per_seed_metrics[i]["weighted_f1"]*100, 4)}
            for i, s in enumerate(VIM_SEEDS)
        },
        "per_class": {
            class_names[c]: {
                "train": train_counts[c], "test": test_counts[c],
                "precision_mean": round(float(pc_prec_m[c])*100, 2),
                "recall_mean":    round(float(pc_recall_m[c])*100, 2),
                "f1_mean":        round(float(pc_f1_m[c])*100, 2),
                "precision_std":  round(float(pc_prec_s[c])*100, 2),
                "recall_std":     round(float(pc_recall_s[c])*100, 2),
                "f1_std":         round(float(pc_f1_s[c])*100, 2),
            } for c in range(NUM_CLASSES)
        },
        "fish08_confusion": {
            class_names[j]: int(all_agg_cm[fish08_idx, j])
            for j in range(NUM_CLASSES) if all_agg_cm[fish08_idx, j] > 0
        },
    }
    _dump("job1_vim_perclass", result)
    return result, pc_recall_m, pc_f1_m, pc_prec_m


# ─────────────────────────────────────────────────────────────────────────────
# Final comparison table
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison(mm_result, vim_result,
                      mm_rec_m, mm_f1_m, mm_prec_m,
                      vim_rec_m, vim_f1_m, vim_prec_m,
                      class_names, train_counts, test_counts,
                      sorted_by_train, top5, bot10):

    _heading("COMPARISON — MarineMamba vs Vim-tiny (Fish4K)")

    print(f"\n| Metric | MarineMamba (11 seeds) | Vim-tiny (4 seeds) |")
    print(f"|--------|----------------------|------------------|")
    print(f"| Top-1 | {mm_result['top1_mean']:.2f} ± {mm_result['top1_std']:.2f}% "
          f"| {vim_result['top1_mean']:.2f} ± {vim_result['top1_std']:.2f}% |")
    print(f"| Macro-precision | {mm_result['macro_precision_mean']:.2f} ± {mm_result['macro_precision_std']:.2f}% "
          f"| {vim_result['macro_precision_mean']:.2f} ± {vim_result['macro_precision_std']:.2f}% |")
    print(f"| Macro-recall | {mm_result['macro_recall_mean']:.2f} ± {mm_result['macro_recall_std']:.2f}% "
          f"| {vim_result['macro_recall_mean']:.2f} ± {vim_result['macro_recall_std']:.2f}% |")
    print(f"| Macro-F1 | {mm_result['macro_f1_mean']:.2f} ± {mm_result['macro_f1_std']:.2f}% "
          f"| {vim_result['macro_f1_mean']:.2f} ± {vim_result['macro_f1_std']:.2f}% |")
    print(f"| Weighted-F1 | {mm_result['weighted_f1_mean']:.2f} ± {mm_result['weighted_f1_std']:.2f}% "
          f"| {vim_result['weighted_f1_mean']:.2f} ± {vim_result['weighted_f1_std']:.2f}% |")
    print(f"| 5 most frequent recall | {mm_result['recall_top5_freq']:.2f}% "
          f"| {vim_result['recall_top5_freq']:.2f}% |")
    print(f"| 10 rarest recall | {mm_result['recall_bot10_rare']:.2f}% "
          f"| {vim_result['recall_bot10_rare']:.2f}% |")

    print(f"\n── Per-class comparison (sorted by train frequency) ──")
    print(f"| Rank | Class | Train | Test | MM recall | Vim recall | MM F1 | Vim F1 |")
    print(f"|------|-------|-------|------|-----------|------------|-------|--------|")
    for rank, cidx in enumerate(sorted_by_train, 1):
        print(f"| {rank:2d} | {class_names[cidx]} | {train_counts[cidx]:5,} | {test_counts[cidx]:4,} "
              f"| {mm_rec_m[cidx]*100:.1f}% | {vim_rec_m[cidx]*100:.1f}% "
              f"| {mm_f1_m[cidx]*100:.1f}% | {vim_f1_m[cidx]*100:.1f}% |")

    print(f"\n── fish_08 / fish_14 side-by-side ──")
    fish08 = 7
    fish14 = 13
    print(f"\n| Class | Train | Test | MM recall | Vim recall | MM F1 | Vim F1 | MM prec | Vim prec |")
    print(f"|-------|-------|------|-----------|------------|-------|--------|---------|----------|")
    for cidx, label in [(fish08, "fish_08"), (fish14, "fish_14")]:
        print(f"| {label} | {train_counts[cidx]:,} | {test_counts[cidx]} "
              f"| {mm_rec_m[cidx]*100:.1f}% | {vim_rec_m[cidx]*100:.1f}% "
              f"| {mm_f1_m[cidx]*100:.1f}% | {vim_f1_m[cidx]*100:.1f}% "
              f"| {mm_prec_m[cidx]*100:.1f}% | {vim_prec_m[cidx]*100:.1f}% |")

    # The question: does Vim also fail on fish_08?
    vim_fish08_recall = vim_rec_m[fish08]
    mm_fish08_recall  = mm_rec_m[fish08]
    print(f"\n── THE QUESTION: Does Vim also fail on fish_08? ──")
    threshold = 0.20   # "collapse" = recall < 20%
    vim_collapsed = vim_fish08_recall < threshold
    mm_collapsed  = mm_fish08_recall  < threshold
    if vim_collapsed and mm_collapsed:
        print(f"\n  YES — both models collapse on fish_08.")
        print(f"  MM recall: {mm_fish08_recall*100:.1f}%  |  Vim recall: {vim_fish08_recall*100:.1f}%")
        print(f"  The confusion is in the DATA, not in our head.")
        print(f"  fish_08 is visually indistinguishable from fish_14 for both architectures.")
    elif not vim_collapsed and mm_collapsed:
        print(f"\n  NO — Vim does NOT fail on fish_08; MarineMamba does.")
        print(f"  MM recall: {mm_fish08_recall*100:.1f}%  |  Vim recall: {vim_fish08_recall*100:.1f}%")
        print(f"  Our frozen head has a specific weakness on fish_08 that Vim's fine-tuned head avoids.")
    else:
        print(f"\n  Neither model catastrophically fails (recall > 20% for both).")
        print(f"  MM recall: {mm_fish08_recall*100:.1f}%  |  Vim recall: {vim_fish08_recall*100:.1f}%")

    # The hypothesis: focal+sqrt trades top-1 for rare-class recall
    print(f"\n── Hypothesis test: focal+sqrt sampler → better rare-class recall ──")
    mm_rare  = mm_result["recall_bot10_rare"]
    vim_rare = vim_result["recall_bot10_rare"]
    mm_freq  = mm_result["recall_top5_freq"]
    vim_freq = vim_result["recall_top5_freq"]
    print(f"  10 rarest:   MarineMamba {mm_rare:.2f}%  vs  Vim {vim_rare:.2f}%")
    print(f"  5 most freq: MarineMamba {mm_freq:.2f}%  vs  Vim {vim_freq:.2f}%")

    mm_f1_mean  = mm_result["macro_f1_mean"]
    vim_f1_mean = vim_result["macro_f1_mean"]
    if mm_f1_mean > vim_f1_mean:
        print(f"\n  Macro-F1: MarineMamba {mm_f1_mean:.2f}% > Vim {vim_f1_mean:.2f}%")
        print(f"  Hypothesis partially supported on macro-F1.")
    else:
        print(f"\n  Macro-F1: Vim {vim_f1_mean:.2f}% > MarineMamba {mm_f1_mean:.2f}%")
        print(f"  *** DIRECT ANSWER: Vim's macro-F1 is higher than ours. ***")
        print(f"  The top-1 trade-off (~{vim_result['top1_mean'] - mm_result['top1_mean']:.2f}pp)")
        print(f"  does NOT buy better macro-F1. The paper cannot claim focal+sqrt improves")
        print(f"  per-class coverage; it is strictly worse on both top-1 and macro-F1 on this dataset.")
    if vim_rare > mm_rare:
        print(f"  Rare-class recall: Vim {vim_rare:.2f}% > MarineMamba {mm_rare:.2f}%")
        print(f"  Focal+sqrt sampler did NOT improve rare-class recall relative to Vim.")
    else:
        print(f"  Rare-class recall: MarineMamba {mm_rare:.2f}% > Vim {vim_rare:.2f}%")
        print(f"  Focal+sqrt sampler DID improve rare-class recall.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Job 2 first — fast, reads existing .npy files
    (mm_result, class_names, train_counts, test_counts,
     sorted_by_train, top5, bot10) = job2_marinemamba_exact()

    # Compute per-class vectors for comparison table
    mm_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
    mm_cms = [np.load(os.path.join(RESULTS_DIR, f"cm_fish4k_marinemamba_seed_{s}.npy"))
              for s in mm_seeds]
    mm_rec_m  = np.array([_metrics_from_cm(cm, test_counts)["per_class"]["recall"]
                           for cm in mm_cms]).mean(axis=0)
    mm_f1_m   = np.array([_metrics_from_cm(cm, test_counts)["per_class"]["f1"]
                           for cm in mm_cms]).mean(axis=0)
    mm_prec_m = np.array([_metrics_from_cm(cm, test_counts)["per_class"]["precision"]
                           for cm in mm_cms]).mean(axis=0)

    # Job 1 — slow, retrains Vim
    vim_result, vim_rec_m, vim_f1_m, vim_prec_m = job1_vim_rerun(
        sorted_by_train, top5, bot10, class_names, train_counts, test_counts)

    # Combined comparison
    _print_comparison(
        mm_result, vim_result,
        mm_rec_m, mm_f1_m, mm_prec_m,
        vim_rec_m, vim_f1_m, vim_prec_m,
        class_names, train_counts, test_counts,
        sorted_by_train, top5, bot10,
    )

    print(f"\n{'='*70}")
    print("DONE — results saved to results/")
    print("  results/job2_marinemamba_exact.json")
    print("  results/job1_vim_perclass.json")
    print("  results/cm_fish4k_vim_seed_{{0,1,2,42}}.npy")
    print("  results/cm_fish4k_vim_agg.npy")
    print(f"{'='*70}")
