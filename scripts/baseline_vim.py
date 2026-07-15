"""
baseline_vim.py — Fine-tune pretrained Vision Mamba (Vim-tiny) on AQUA20, Sea23, Fish4K.

Trains each (dataset, seed) pair, saves the best checkpoint, then evaluates
macro-F1 / weighted-F1 / precision / recall from the saved checkpoint.
Resumes automatically: if a checkpoint already exists for a seed, training is
skipped and only the evaluation step runs.

Outputs
-------
  results_vim_{dataset}.json         — top-1 summary (original format, backward-compat)
  results/vim_{dataset}_f1.json      — full metrics incl. macro-F1, per-seed breakdown
  results/best_model_vim_{dataset}_seed_{N}.pth  — best checkpoint per seed
  logs/log_vim_{dataset}_seed_{N}.txt            — per-seed training log

Vision Mamba reference:
    Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with
    Bidirectional State Space Model", ICML 2024.
    https://arxiv.org/abs/2401.13586

Pretrained weights (ImageNet-1k, 76.1% top-1):
    https://huggingface.co/hustvl/Vim-tiny-midclstok

Usage:
    python3 scripts/baseline_vim.py --dataset aqua20 --seeds 0 1 2 42
    python3 scripts/baseline_vim.py --dataset sea23  --seeds 0 1 2 3 4
    python3 scripts/baseline_vim.py --dataset fish4k --seeds 0 1 2
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from marinemamba.paths import AQUA20_ROOT, SEA23_ROOT, FISH4K_ROOT, LOGS_DIR
from marinemamba.utils import set_seed, ensure_dir

##  ---── Vim model cache ───────────────────────────────────────────────────────────

_FINETUNING_VIM = Path(
    "/localhome/ehoseinz/PycharmProjects/EEG/finetuning vimamba/Vim"
)
VIM_CACHE = (
    _FINETUNING_VIM / "vim"
    if _FINETUNING_VIM.exists()
    else PROJECT_ROOT / "data" / "_vim_repo" / "vim"
)
VIM_REPO = "https://github.com/hustvl/Vim.git"
VIM_CKPT = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/"
    "models--hustvl--Vim-tiny-midclstok/snapshots/"
    "07c00e0e4ea2973d8e343afdd807128a57bc9fd5/"
    "vim_t_midclstok_76p1acc.pth"
))

RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_CFG = {
    "aqua20": {"path": Path(AQUA20_ROOT), "num_classes": 20},
    "sea23":  {"path": Path(SEA23_ROOT),  "num_classes": 23},
    "fish4k": {"path": Path(FISH4K_ROOT), "num_classes": 23},
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── Model ──────────────────────────────────────────────────────────────────────

def _ensure_vim_repo():
    if VIM_CACHE.exists():
        return
    print("Cloning Vim repository (one-time setup)...")
    VIM_CACHE.parent.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", VIM_REPO, str(VIM_CACHE.parent)],
        check=True,
    )


def _ensure_vim_weights():
    if VIM_CKPT.exists():
        return
    print("Downloading Vim-tiny pretrained weights from Hugging Face...")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="hustvl/Vim-tiny-midclstok",
            filename="vim_t_midclstok_76p1acc.pth",
        )
    except Exception as e:
        print(f"ERROR: Could not download Vim weights: {e}")
        print("Download manually from: https://huggingface.co/hustvl/Vim-tiny-midclstok")
        sys.exit(1)


def build_vim(num_classes: int, device: str) -> nn.Module:
    _ensure_vim_repo()
    _ensure_vim_weights()

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
    assert set(missing) == {"head.weight", "head.bias"}, \
        f"Unexpected missing keys: {missing}"
    assert not unexpected, f"Unexpected extra keys: {unexpected}"

    total = sum(p.numel() for p in model.parameters())
    print(f"  Vim-tiny loaded  ({total/1e6:.2f}M params, full fine-tune)")
    return model.to(device)


# ── Data ───────────────────────────────────────────────────────────────────────

class _SafeImageFolder(datasets.ImageFolder):
    """Skip corrupted images instead of crashing."""
    def __getitem__(self, index):
        for _ in range(10):
            try:
                return super().__getitem__(index)
            except Exception:
                index = random.randint(0, len(self) - 1)
        return transforms.ToTensor()(Image.new("RGB", (224, 224))), 0


def build_loaders(dataset_path: Path, batch_size: int):
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
    trainset = _SafeImageFolder(str(dataset_path / "train"), transform=train_tf)
    testset  = _SafeImageFolder(str(dataset_path / "test"),  transform=val_tf)
    kw = dict(num_workers=0, pin_memory=True)
    return (DataLoader(trainset, batch_size=batch_size, shuffle=True,  **kw),
            DataLoader(testset,  batch_size=batch_size, shuffle=False, **kw),
            testset.classes)


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_seed(dataset: str, seed: int, cfg: dict, args) -> tuple[float, Path, list]:
    """
    Train Vim-tiny for one (dataset, seed).

    Returns (best_top1_acc, checkpoint_path, class_names).
    If a checkpoint already exists the training phase is skipped.
    """
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = RESULTS_DIR / f"best_model_vim_{dataset}_seed_{seed}.pth"

    # Resume: skip training if checkpoint already exists
    if ckpt_path.exists():
        log_path = os.path.join(LOGS_DIR, f"log_vim_{dataset}_seed_{seed}.txt")
        best_acc = 0.0
        if os.path.exists(log_path):
            with open(log_path) as lf:
                for line in lf:
                    if line.startswith("\nFinal best accuracy:"):
                        try:
                            best_acc = float(line.split(":")[1].strip().rstrip("%"))
                        except Exception:
                            pass
        _, _, class_names = build_loaders(cfg["path"], args.batch_size)
        print(f"  [resume] checkpoint found — skipping training  best_acc={best_acc:.4f}%")
        return best_acc, ckpt_path, class_names

    set_seed(seed)
    ensure_dir(LOGS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"log_vim_{dataset}_seed_{seed}.txt")

    class _Tee:
        def __init__(self, path):
            self.f = open(path, "w")
        def write(self, m):
            self.f.write(m); self.f.flush(); sys.__stdout__.write(m)
        def flush(self):
            self.f.flush(); sys.__stdout__.flush()

    sys.stdout = _Tee(log_path)
    try:
        print(f"Vim-tiny | dataset={dataset} seed={seed} device={device}")
        print(f"epochs={args.epochs} lr={args.lr} batch={args.batch_size} "
              f"warmup={args.warmup_epochs} patience={args.patience}")

        train_loader, test_loader, class_names = build_loaders(cfg["path"], args.batch_size)
        print(f"Train={len(train_loader.dataset)}  Test={len(test_loader.dataset)}  "
              f"Classes={cfg['num_classes']}")

        model     = build_vim(cfg["num_classes"], device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

        total_steps  = args.epochs * len(train_loader)
        warmup_steps = args.warmup_epochs * len(train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * t))

        scheduler  = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        best_acc   = 0.0
        no_improve = 0

        for epoch in range(args.epochs):
            t0 = time.time()
            model.train()
            tr_loss = tr_correct = tr_total = 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
                optimizer.step()
                scheduler.step()
                tr_loss    += loss.item()
                tr_correct += out.argmax(1).eq(labels).sum().item()
                tr_total   += labels.size(0)

            model.eval()
            te_correct = te_total = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    out = model(imgs)
                    te_correct += out.argmax(1).eq(labels).sum().item()
                    te_total   += labels.size(0)

            test_acc = 100.0 * te_correct / te_total
            lr_now   = optimizer.param_groups[0]["lr"]
            print(f"Ep {epoch+1:03d} | {time.time()-t0:.1f}s | lr={lr_now:.2e} | "
                  f"train {tr_loss/len(train_loader):.4f}/{100*tr_correct/tr_total:.2f}% | "
                  f"test {test_acc:.2f}%")

            if test_acc > best_acc:
                best_acc   = test_acc
                no_improve = 0
                torch.save(model.state_dict(), str(ckpt_path))
                print(f"  -> best={best_acc:.4f}%  (saved)")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"\nFinal best accuracy: {best_acc:.4f}%")
    finally:
        sys.stdout = sys.__stdout__

    return best_acc, ckpt_path, class_names


# ── Evaluation ─────────────────────────────────────────────────────────────────

def eval_ckpt(ckpt_path: Path, cfg: dict, class_names: list, batch_size: int) -> dict:
    """Load checkpoint and compute top-1 / macro-F1 / weighted-F1 on the test set."""
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = cfg["num_classes"]

    model = build_vim(num_classes, device)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=False))
    model.eval()

    _, test_loader, _ = build_loaders(cfg["path"], batch_size)
    test_dir    = cfg["path"] / "test"
    test_counts = [len(list((test_dir / c).iterdir())) for c in class_names]

    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            out = model(imgs.to(device))
            all_pred.extend(out.argmax(1).cpu().tolist())
            all_true.extend(labels.tolist())

    top1 = 100.0 * sum(p == t for p, t in zip(all_pred, all_true)) / len(all_true)
    cm   = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))
    tp   = cm.diagonal().astype(float)
    fp   = cm.sum(0) - tp
    fn   = cm.sum(1) - tp
    prec   = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1_pc  = np.where(prec + recall > 0, 2 * prec * recall / (prec + recall), 0.0)
    support = np.array(test_counts, dtype=float)
    w_f1    = (f1_pc * support).sum() / support.sum() * 100

    return {
        "top1":         round(top1,               4),
        "macro_prec":   round(prec.mean()   * 100, 2),
        "macro_recall": round(recall.mean() * 100, 2),
        "macro_f1":     round(f1_pc.mean()  * 100, 2),
        "weighted_f1":  round(w_f1,               2),
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Fine-tune Vim-tiny on AQUA20 / Sea23 / Fish4K with macro-F1 evaluation"
    )
    p.add_argument("--datasets", nargs="+", default=list(DATASET_CFG),
                   choices=list(DATASET_CFG), dest="datasets")
    p.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2, 42])
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=5e-5)
    p.add_argument("--batch-size",    type=int,   default=64,  dest="batch_size")
    p.add_argument("--warmup-epochs", type=int,   default=5,   dest="warmup_epochs")
    p.add_argument("--patience",      type=int,   default=20)
    args = p.parse_args()

    print(f"\nVim-tiny  |  Datasets: {args.datasets}  |  Seeds: {args.seeds}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

    for dataset in args.datasets:
        cfg          = DATASET_CFG[dataset]
        top1_results = {}
        seed_metrics = {}
        class_names  = None

        print(f"\n{'#'*64}")
        print(f"# {dataset.upper()}  ({cfg['num_classes']} classes)")
        print(f"{'#'*64}")

        for seed in args.seeds:
            print(f"\n{'='*60}\nSeed {seed} — {dataset}\n{'='*60}")

            best_acc, ckpt_path, class_names = train_one_seed(dataset, seed, cfg, args)

            print(f"  Evaluating checkpoint for macro-F1 ...")
            metrics = eval_ckpt(ckpt_path, cfg, class_names, args.batch_size)
            seed_metrics[seed]      = metrics
            top1_results[str(seed)] = round(best_acc, 4)

            print(f"  top-1={metrics['top1']:.2f}%  "
                  f"macro-F1={metrics['macro_f1']:.2f}%  "
                  f"weighted-F1={metrics['weighted_f1']:.2f}%")

        # ── Aggregate ──────────────────────────────────────────────────────────

        def _ms(key):
            vals = [seed_metrics[s][key] for s in args.seeds]
            return round(float(np.mean(vals)), 2), round(float(np.std(vals)), 2)

        t1_m, t1_s = _ms("top1")
        mp_m, mp_s = _ms("macro_prec")
        mr_m, mr_s = _ms("macro_recall")
        mf_m, mf_s = _ms("macro_f1")
        wf_m, wf_s = _ms("weighted_f1")

        print(f"\n{'='*60}")
        print(f"Vim-tiny — {dataset} — seeds {args.seeds}")
        print(f"{'='*60}")
        print(f"| Metric          | Mean ± Std          |")
        print(f"|-----------------|---------------------|")
        print(f"| Top-1           | {t1_m:.2f} ± {t1_s:.2f}%  |")
        print(f"| Macro-precision | {mp_m:.2f} ± {mp_s:.2f}%  |")
        print(f"| Macro-recall    | {mr_m:.2f} ± {mr_s:.2f}%  |")
        print(f"| Macro-F1        | {mf_m:.2f} ± {mf_s:.2f}%  |")
        print(f"| Weighted-F1     | {wf_m:.2f} ± {wf_s:.2f}%  |")
        print(f"\n| Seed | Top-1  | Macro-F1 | Weighted-F1 |")
        print(f"|------|--------|----------|-------------|")
        for s in args.seeds:
            r = seed_metrics[s]
            print(f"| {s:<4} | {r['top1']:.2f}% | {r['macro_f1']:.2f}%   | {r['weighted_f1']:.2f}%   |")

        # ── Save results ───────────────────────────────────────────────────────

        top1_vals    = list(top1_results.values())
        top1_summary = {
            "seeds": top1_results,
            "mean":  round(float(np.mean(top1_vals)), 4),
            "std":   round(float(np.std(top1_vals)),  4),
        }
        compat_path = PROJECT_ROOT / f"results_vim_{dataset}.json"
        with open(compat_path, "w") as f:
            json.dump(top1_summary, f, indent=2)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        full_result = {
            "model": "Vim-tiny", "dataset": dataset, "seeds": args.seeds,
            "top1_mean":            t1_m, "top1_std":            t1_s,
            "macro_precision_mean": mp_m, "macro_precision_std": mp_s,
            "macro_recall_mean":    mr_m, "macro_recall_std":    mr_s,
            "macro_f1_mean":        mf_m, "macro_f1_std":        mf_s,
            "weighted_f1_mean":     wf_m, "weighted_f1_std":     wf_s,
            "per_seed": {
                str(s): {k: seed_metrics[s][k]
                         for k in ("top1", "macro_prec", "macro_recall",
                                   "macro_f1", "weighted_f1")}
                for s in args.seeds
            },
        }
        f1_path = RESULTS_DIR / f"vim_{dataset}_f1.json"
        with open(f1_path, "w") as f:
            json.dump(full_result, f, indent=2)

        print(f"\n→ {compat_path}  (top-1 summary)")
        print(f"→ {f1_path}  (full metrics)")


if __name__ == "__main__":
    main()
