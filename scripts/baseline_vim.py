"""
baseline_vim.py — Fine-tune pretrained Vision Mamba (Vim-tiny) on AQUA20, Sea23, Fish4K.

Used as a comparison baseline against PyramidCLIPSpyMamba.

Vision Mamba reference:
    Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with
    Bidirectional State Space Model", ICML 2024.
    https://arxiv.org/abs/2401.13586

Pretrained weights (ImageNet-1k, 76.1% top-1):
    https://huggingface.co/hustvl/Vim-tiny-midclstok

Usage:
    python3 scripts/baseline_vim.py --dataset aqua20 --seeds 0 1 2 42
    python3 scripts/baseline_vim.py --dataset sea23  --seeds 0 1 2 42
    python3 scripts/baseline_vim.py --dataset fish4k --seeds 0 1 2 42
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from spymamba.paths import AQUA20_ROOT, SEA23_ROOT, FISH4K_ROOT, LOGS_DIR
from spymamba.utils import set_seed, ensure_dir

# ── Vim model cache ───────────────────────────────────────────────────────────

# Prefer the existing working clone (older commit whose models_mamba.py defines
# Mamba locally, compatible with the installed mamba_ssm / causal-conv1d).
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
        check=True
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
        print("Download manually from:")
        print("  https://huggingface.co/hustvl/Vim-tiny-midclstok")
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

    # Load ImageNet pretrained weights; skip head (shape mismatch 1000 → num_classes)
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
    # num_workers=0 avoids segfaults from corrupted JPEG files in worker processes
    kw = dict(num_workers=0, pin_memory=True)
    return (DataLoader(trainset, batch_size=batch_size, shuffle=True,  **kw),
            DataLoader(testset,  batch_size=batch_size, shuffle=False, **kw),
            trainset.classes)


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_seed(dataset: str, seed: int, cfg: dict, args) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    ensure_dir(LOGS_DIR)
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
        print(f"epochs={args.epochs} lr={args.lr} batch={args.batch_size} patience={args.patience}")

        train_loader, test_loader, classes = build_loaders(cfg["path"], args.batch_size)
        print(f"Train={len(train_loader.dataset)}  Test={len(test_loader.dataset)}  Classes={len(classes)}")

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                print(f"  -> best={best_acc:.4f}%")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"\nFinal best accuracy: {best_acc:.4f}%")
    finally:
        sys.stdout = sys.__stdout__
    return best_acc


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Fine-tune Vim-tiny on AQUA20 / Sea23 / Fish4K"
    )
    p.add_argument("--dataset",       choices=list(DATASET_CFG), required=True)
    p.add_argument("--seeds",         type=int, nargs="+", default=[0, 1, 2, 42])
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=5e-5)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--warmup-epochs", type=int,   default=5,  dest="warmup_epochs")
    p.add_argument("--patience",      type=int,   default=20)
    args = p.parse_args()

    cfg     = DATASET_CFG[args.dataset]
    results = {}

    for seed in args.seeds:
        print(f"\n{'='*60}\nSeed {seed} — {args.dataset}\n{'='*60}")
        acc = train_one_seed(args.dataset, seed, cfg, args)
        results[str(seed)] = round(acc, 4)
        print(f"Seed {seed}: {acc:.4f}%")

    vals = list(results.values())
    summary = {
        "seeds": results,
        "mean":  round(float(np.mean(vals)), 4),
        "std":   round(float(np.std(vals)),  4),
    }

    out_path = PROJECT_ROOT / f"results_vim_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Vim-tiny — {args.dataset}")
    print(f"  Seeds : {results}")
    print(f"  Mean  : {summary['mean']:.4f}%  Std: {summary['std']:.4f}%")
    print(f"  Saved : {out_path}")


if __name__ == "__main__":
    main()
