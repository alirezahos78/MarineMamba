"""
Ablation study on AQUA20 — scan order and model components.

Variants:
  spiral        — full model with spiral scan (baseline)
  raster        — full model with raster (row-major) scan
  no_focal_loss — CrossEntropyLoss instead of Focal Loss
  no_mamba      — token-wise FFN block instead of Mamba SSM
  no_cls        — no CLS token injected into branches
  fine_only     — single ViT-B/16 (14×14) branch
  coarse_only   — single ViT-B/32 (7×7) branch

Logs   : logs/log_ablation_{variant}_seed_{N}.txt
Results: ablation_results.json

Usage:
    python3 scripts/ablation.py
    python3 scripts/ablation.py --variants spiral raster
    python3 scripts/ablation.py --variants no_mamba no_cls --seeds 0 1 2
"""

import argparse
import copy
import gc
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
from einops import rearrange
from tqdm import tqdm

from marinemamba.config import get_config
from marinemamba.data import build_dataloaders
from marinemamba.losses import build_criterion
from marinemamba.model import PyramidBranch, MarineMamba, MarineMambaBlock, SCAtt
from marinemamba.paths import LOGS_DIR, PROJECT_ROOT, RESULTS_PATH
from marinemamba.trainer import build_model
from marinemamba.utils import Logger, ensure_dir, set_seed

CONFIG_NAME  = "aqua20_dual_hybrid_128_focal_balanced"
RESULTS_FILE = os.path.join(PROJECT_ROOT, "ablation_results.json")


# ---------------------------------------------------------------------------
# Scan variants
# ---------------------------------------------------------------------------

class RasterScanner(nn.Module):
    """Row-major scanner — drop-in replacement for SpiralScanner."""
    def __init__(self, height, width):
        super().__init__()
        n = height * width
        idx_fwd = torch.arange(n, dtype=torch.long)
        self.register_buffer("idx_fwd", idx_fwd)
        self.register_buffer("inv_fwd", torch.argsort(idx_fwd))
        self.register_buffer("idx_bwd", idx_fwd.flip(0).clone())
        self.register_buffer("inv_bwd", torch.argsort(idx_fwd.flip(0).clone()))

    def forward(self, x):
        return x[:, self.idx_fwd, :], x[:, self.idx_bwd, :]

    def inverse(self, fwd, bwd):
        return fwd[:, self.inv_fwd, :], bwd[:, self.inv_bwd, :]


# ---------------------------------------------------------------------------
# Model-component variants
# ---------------------------------------------------------------------------

class FFNBlock(nn.Module):
    """Token-wise FFN + CBAM — replaces MarineMambaBlock (isolates Mamba contribution)."""
    def __init__(self, dim, height, width, ffn_drop=0.0, layer_scale_init=1.0):
        super().__init__()
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

    def forward(self, x):
        B, L, C = x.shape
        has_cls  = (L == self.num_spatial + 1)
        residual = x
        h        = self.norm1(x)
        spatial  = h[:, 1:, :] if has_cls else h
        y_2d     = rearrange(spatial, "b (h w) c -> b c h w", h=self.height, w=self.width)
        y_2d     = self.sc_att(y_2d)
        y_sp     = rearrange(y_2d, "b c h w -> b (h w) c")
        y = torch.cat([h[:, :1, :], y_sp], dim=1) if has_cls else y_sp
        x = residual + self.ls1 * y
        x = x + self.ls2 * self.ffn(self.norm2(x))
        return x


class SingleBranchCLIPMamba(nn.Module):
    """Full model reduced to one branch."""
    def __init__(self, branch, use_fine, dim, num_classes, dropout=0.3, head_hidden_dim=512):
        super().__init__()
        self.branch   = branch
        self.use_fine = use_fine
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, fine_feats, coarse_feats, fine_cls, coarse_cls):
        feats = self.branch(fine_feats, fine_cls) if self.use_fine \
                else self.branch(coarse_feats, coarse_cls)
        return self.head(feats)


# ---------------------------------------------------------------------------
# Builders — all return (model, effective_config)
# ---------------------------------------------------------------------------

def build_spiral(config, device):
    return build_model(config, device), config


def build_raster(config, device):
    model = build_model(config, device)
    for block in model.modules():
        if isinstance(block, MarineMambaBlock):
            block.scanner = RasterScanner(block.height, block.width).to(device)
    return model, config


def build_no_focal(config, device):
    cfg = copy.deepcopy(config)
    cfg["loss"] = "ce"
    return build_model(cfg, device), cfg


def build_no_mamba(config, device):
    model = build_model(config, device)
    dim   = config["dim"]
    for branch in (model.fine_branch, model.coarse_branch):
        branch.blocks = nn.ModuleList([
            FFNBlock(dim=dim, height=blk.height, width=blk.width,
                     ffn_drop=config.get("ffn_drop", 0.0),
                     layer_scale_init=config.get("layer_scale_init", 1.0))
            for blk in branch.blocks
        ]).to(device)
    return model.to(device), config


def _no_cls_branch_forward(self, x, cls):
    B = x.shape[0]
    x = x.permute(0, 2, 3, 1).reshape(B, -1, x.shape[1])
    x = self.input_projection(x)
    x = x + self.pos_embed[:, 1:, :]
    for block in self.blocks:
        x = block(x)
    return self.norm(x).mean(dim=1)


def build_no_cls(config, device):
    model = build_model(config, device)
    for branch in (model.fine_branch, model.coarse_branch):
        branch.forward = types.MethodType(_no_cls_branch_forward, branch)
    return model, config


def build_fine_only(config, device):
    full  = build_model(config, device)
    model = SingleBranchCLIPMamba(
        branch=full.fine_branch, use_fine=True,
        dim=config["dim"], num_classes=config["num_classes"],
        dropout=config.get("dropout", 0.3),
        head_hidden_dim=config.get("head_hidden_dim", 512),
    ).to(device)
    return model, config


def build_coarse_only(config, device):
    full  = build_model(config, device)
    model = SingleBranchCLIPMamba(
        branch=full.coarse_branch, use_fine=False,
        dim=config["dim"], num_classes=config["num_classes"],
        dropout=config.get("dropout", 0.3),
        head_hidden_dim=config.get("head_hidden_dim", 512),
    ).to(device)
    return model, config


BUILDERS = {
    "spiral":       build_spiral,
    "raster":       build_raster,
    "no_focal_loss": build_no_focal,
    "no_mamba":     build_no_mamba,
    "no_cls":       build_no_cls,
    "fine_only":    build_fine_only,
    "coarse_only":  build_coarse_only,
}

ALL_VARIANTS = list(BUILDERS)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(model, config, run_name, device):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.3f} M")

    trainloader, testloader = build_dataloaders(config)
    print(f"  Train: {len(trainloader.dataset)}  |  Test: {len(testloader.dataset)}")

    train_labels = trainloader.dataset.labels
    class_counts = torch.bincount(train_labels, minlength=config["num_classes"])
    criterion    = build_criterion(config, config["num_classes"], class_counts, device)
    optimizer    = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    total_steps  = config["epochs"] * len(trainloader)
    warmup_steps = config["warmup_epochs"] * len(trainloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        t = step - warmup_steps
        return 0.5 * (1.0 + math.cos(math.pi * t / max(1, total_steps - warmup_steps)))

    scheduler  = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_acc   = 0.0
    no_improve = 0
    patience   = config.get("early_stopping_patience")
    nb         = device == "cuda"

    for epoch in range(config["epochs"]):
        model.train()
        correct_tr = total_tr = 0
        train_loss = 0.0

        for batch in tqdm(trainloader, desc=f"Ep {epoch+1}/{config['epochs']}", leave=False, file=sys.stderr):
            fine_feats, coarse_feats, fine_cls, coarse_cls, targets = batch
            fine_feats   = fine_feats.to(device,  non_blocking=nb)
            coarse_feats = coarse_feats.to(device, non_blocking=nb)
            fine_cls     = fine_cls.to(device,     non_blocking=nb)
            coarse_cls   = coarse_cls.to(device,   non_blocking=nb)
            targets      = targets.to(device,      non_blocking=nb)

            optimizer.zero_grad(set_to_none=True)
            out  = model(fine_feats, coarse_feats, fine_cls, coarse_cls)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            correct_tr += out.argmax(1).eq(targets).sum().item()
            train_loss += loss.item()
            total_tr   += targets.size(0)

        model.eval()
        correct_te = total_te = 0
        with torch.inference_mode():
            for batch in testloader:
                fine_feats, coarse_feats, fine_cls, coarse_cls, targets = batch
                fine_feats   = fine_feats.to(device,  non_blocking=nb)
                coarse_feats = coarse_feats.to(device, non_blocking=nb)
                fine_cls     = fine_cls.to(device,     non_blocking=nb)
                coarse_cls   = coarse_cls.to(device,   non_blocking=nb)
                targets      = targets.to(device,      non_blocking=nb)
                out = model(fine_feats, coarse_feats, fine_cls, coarse_cls)
                correct_te += out.argmax(1).eq(targets).sum().item()
                total_te   += targets.size(0)

        test_acc = 100.0 * correct_te / total_te
        avg_loss = train_loss / len(trainloader)

        if test_acc > best_acc:
            best_acc   = test_acc
            no_improve = 0
            print(f"New best: {best_acc:.2f}%")
        else:
            no_improve += 1

        print(f"{run_name} | Ep {epoch+1:02d} | Top1: {test_acc:.2f}%  "
              f"Best: {best_acc:.2f}%  Loss: {avg_loss:.4f}  "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if patience and no_improve >= patience:
            print(f"Early stopping — best: {best_acc:.2f}%")
            break

    gc.collect()
    torch.cuda.empty_cache() if device == "cuda" else None
    return best_acc


def run_variant(name, builder, base_config, seeds, device):
    ensure_dir(LOGS_DIR)
    per_seed = {}
    for seed in seeds:
        run_name = f"ablation_{name}_seed_{seed}"
        log_path = os.path.join(LOGS_DIR, f"log_{run_name}.txt")

        print(f"\n{'='*60}")
        print(f"Variant: {name}  |  Seed: {seed}")
        print(f"{'='*60}")

        orig = sys.stdout
        logger = Logger(log_path)
        sys.stdout = logger
        try:
            set_seed(seed)
            model, config = builder(base_config, device)
            acc = _train(model, config, run_name, device)
        finally:
            sys.stdout = orig
            logger.close()

        per_seed[seed] = acc
        print(f"  {name:<16}  seed {seed}  ->  {acc:.2f}%")

    return per_seed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Ablation study on AQUA20")
    p.add_argument("--seeds",    nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--variants", nargs="+", default=ALL_VARIANTS, choices=ALL_VARIANTS)
    args = p.parse_args()

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    base_config = get_config(CONFIG_NAME)

    print(f"\nAblation — AQUA20  |  Seeds: {args.seeds}  |  Device: {device}")
    print(f"Variants: {args.variants}\n")

    # Load existing results to allow partial re-runs
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    for name in args.variants:
        per_seed = run_variant(name, BUILDERS[name], base_config, args.seeds, device)
        accs = list(per_seed.values())
        results[name] = {
            "seeds":    {str(k): v for k, v in per_seed.items()},
            "mean":     round(float(np.mean(accs)), 4),
            "std":      round(float(np.std(accs)),  4),
            "variance": round(float(np.var(accs)),  4),
        }
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # Load full-model baseline for comparison
    baseline_mean = baseline_std = None
    try:
        with open(RESULTS_PATH) as f:
            saved = json.load(f)
        seed_vals = [saved["aqua20_dual_hybrid_128_focal_balanced"]["seeds"].get(str(s))
                     for s in args.seeds]
        seed_vals = [v for v in seed_vals if v is not None]
        if seed_vals:
            baseline_mean = round(float(np.mean(seed_vals)), 4)
            baseline_std  = round(float(np.std(seed_vals)),  4)
    except Exception:
        pass

    # Print comparison table
    print(f"\n{'='*64}")
    print("ABLATION RESULTS — AQUA20")
    print(f"{'='*64}")
    print(f"{'Variant':<16}  {'Mean':>8}  {'Std':>8}  {'vs full':>10}")
    print(f"{'-'*50}")
    if baseline_mean is not None:
        print(f"{'full (baseline)':<16}  {baseline_mean:>7.2f}%  {baseline_std:>7.4f}  {'—':>10}")
    for name, r in results.items():
        delta = f"{r['mean'] - baseline_mean:+.2f}%" if baseline_mean is not None else ""
        print(f"{name:<16}  {r['mean']:>7.2f}%  {r['std']:>7.4f}  {delta:>10}")

    print(f"\nResults saved to ablation_results.json")


if __name__ == "__main__":
    main()
