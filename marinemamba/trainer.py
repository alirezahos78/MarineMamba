import gc
import json
import math
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from .config import get_config
from .data import build_dataloaders
from .losses import build_criterion
from .model import MarineMamba
from .paths import LOGS_DIR, PROJECT_ROOT, RESULTS_PATH
from .utils import Logger, ensure_dir, set_seed

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def build_model(config, device):
    model = MarineMamba(
        fine_input_dim=config["fine_input_dim"],
        fine_h=config["fine_h"],
        fine_w=config["fine_w"],
        coarse_input_dim=config["coarse_input_dim"],
        coarse_h=config["coarse_h"],
        coarse_w=config["coarse_w"],
        dim=config["dim"],
        fine_depth=config["fine_depth"],
        coarse_depth=config["coarse_depth"],
        num_classes=config["num_classes"],
        cls_dim=config.get("cls_dim", 512),
        stochastic_depth=config.get("stochastic_depth", 0.0),
        ffn_drop=config.get("ffn_drop", 0.0),
        layer_scale_init=config.get("layer_scale_init", 1.0),
        dropout=config.get("dropout", 0.3),
        head_hidden_dim=config.get("head_hidden_dim", 512),
    ).to(device)
    return model


def _make_run_name(config_name, seed):
    return f"{config_name}_seed_{seed}"


def train(config_name, seed):
    config   = get_config(config_name)
    run_name = _make_run_name(config_name, seed)
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 60}")
    print(f"Config : {config['name']}")
    print(f"Device : {device}  |  Seed: {seed}")
    print(f"{'=' * 60}")

    ensure_dir(LOGS_DIR)
    original_stdout = sys.stdout
    logger = Logger(os.path.join(LOGS_DIR, f"log_{run_name}.txt"))
    sys.stdout = logger

    try:
        result = _train_logged(config, run_name, seed, device)
    finally:
        sys.stdout = original_stdout
        logger.close()

    return result


def _train_logged(config, run_name, seed, device):
    set_seed(seed)
    model = build_model(config, device)

    total_params  = sum(p.numel() for p in model.parameters())
    fine_params   = sum(p.numel() for p in model.fine_branch.parameters())
    coarse_params = sum(p.numel() for p in model.coarse_branch.parameters())
    print(f"\nModel parameters: {total_params / 1e6:.3f} M total "
          f"(fine {fine_params / 1e6:.3f} M  |  coarse {coarse_params / 1e6:.3f} M)")

    print("\nLoading dataset...")
    trainloader, testloader = build_dataloaders(config)
    print(f"  Train: {len(trainloader.dataset)} samples  |  Test: {len(testloader.dataset)} samples")

    train_labels = trainloader.dataset.labels if hasattr(trainloader.dataset, "labels") else None
    class_counts = torch.bincount(train_labels, minlength=config["num_classes"]) if train_labels is not None else None
    criterion    = build_criterion(config, config["num_classes"], class_counts, device)

    optimizer    = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    total_steps  = config["epochs"] * len(trainloader)
    warmup_steps = config["warmup_epochs"] * len(trainloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        t     = step - warmup_steps
        t_max = total_steps - warmup_steps
        return 0.5 * (1.0 + math.cos(math.pi * t / t_max))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_acc  = 0.0
    epochs_without_improvement = 0
    patience  = config.get("early_stopping_patience")
    nb        = device == "cuda"

    for epoch in range(config["epochs"]):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        loop = tqdm(trainloader, desc=f"Ep {epoch + 1}/{config['epochs']}", leave=False, file=sys.stderr)
        for batch in loop:
            fine_feats, coarse_feats, fine_cls, coarse_cls, targets = batch
            fine_feats   = fine_feats.to(device,   non_blocking=nb)
            coarse_feats = coarse_feats.to(device,  non_blocking=nb)
            fine_cls     = fine_cls.to(device,      non_blocking=nb)
            coarse_cls   = coarse_cls.to(device,    non_blocking=nb)
            targets      = targets.to(device,       non_blocking=nb)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(fine_feats, coarse_feats, fine_cls, coarse_cls)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            _, predicted = outputs.max(1)
            correct_train += predicted.eq(targets).sum().item()
            total_train   += targets.size(0)
            train_loss    += loss.item()
            loop.set_postfix(loss=loss.item())

        train_acc = 100.0 * correct_train / total_train
        avg_loss  = train_loss / len(trainloader)

        model.eval()
        correct_test = {1: 0, 2: 0, 3: 0}
        total_test   = 0
        with torch.inference_mode():
            for batch in testloader:
                fine_feats, coarse_feats, fine_cls, coarse_cls, targets = batch
                fine_feats   = fine_feats.to(device,   non_blocking=nb)
                coarse_feats = coarse_feats.to(device,  non_blocking=nb)
                fine_cls     = fine_cls.to(device,      non_blocking=nb)
                coarse_cls   = coarse_cls.to(device,    non_blocking=nb)
                targets      = targets.to(device,       non_blocking=nb)
                outputs      = model(fine_feats, coarse_feats, fine_cls, coarse_cls)
                topk = outputs.topk(3, dim=1).indices
                lbl  = targets.unsqueeze(1)
                total_test += targets.size(0)
                correct_test[1] += topk[:, :1].eq(lbl).any(1).sum().item()
                correct_test[2] += topk[:, :2].eq(lbl).any(1).sum().item()
                correct_test[3] += topk[:, :3].eq(lbl).any(1).sum().item()

        test_acc  = 100.0 * correct_test[1] / total_test
        test_acc2 = 100.0 * correct_test[2] / total_test
        test_acc3 = 100.0 * correct_test[3] / total_test

        if test_acc > best_acc:
            best_acc = test_acc
            epochs_without_improvement = 0
            ckpt = os.path.join(PROJECT_ROOT, f"best_model_{run_name}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"New best: {best_acc:.2f}%")
        else:
            epochs_without_improvement += 1

        print(
            f"{config['name']} | Ep {epoch + 1:02d} | "
            f"Train: {train_acc:.2f}%  "
            f"Top1: {test_acc:.2f}%  Top2: {test_acc2:.2f}%  Top3: {test_acc3:.2f}%  "
            f"Best: {best_acc:.2f}%  Loss: {avg_loss:.4f}  LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if patience and epochs_without_improvement >= patience:
            print(f"Early stopping — best: {best_acc:.2f}%")
            break

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "name":         config["name"],
        "seed":         seed,
        "accuracy":     best_acc,
        "params_m":     total_params / 1e6,
        "fine_depth":   config["fine_depth"],
        "coarse_depth": config["coarse_depth"],
        "dim":          config["dim"],
        "num_classes":  config["num_classes"],
    }


def run_multi_seed(config_names=None, seeds=None):
    """
    Train each config across all seeds, then report mean and variance.

    Results are written to results.json as:
        {config_name: {per_seed: {seed: acc}, mean: float, std: float, variance: float}}
    """
    config_names = config_names or ["aqua20_dual_hybrid_128_focal_balanced"]
    seeds        = seeds or SEEDS

    all_results = {}

    for config_name in config_names:
        print(f"\n{'=' * 70}")
        print(f"Config: {config_name}  |  Seeds: {seeds}")
        print(f"{'=' * 70}")

        per_seed = {}
        for seed in seeds:
            result = train(config_name, seed)
            per_seed[seed] = result["accuracy"]
            print(f"  Seed {seed:3d} -> {result['accuracy']:.4f}%")

        accs     = list(per_seed.values())
        mean     = float(np.mean(accs))
        std      = float(np.std(accs))
        variance = float(np.var(accs))

        all_results[config_name] = {
            "name":     get_config(config_name)["name"],
            "seeds":    seeds,
            "per_seed": {str(s): round(a, 4) for s, a in per_seed.items()},
            "mean":     round(mean, 4),
            "std":      round(std, 4),
            "variance": round(variance, 4),
            "params_m": result["params_m"],
            "num_classes": result["num_classes"],
        }

        print(f"\n  {config_name}")
        print(f"  Accuracy : {mean:.2f}% +/- {std:.2f}%  (var={variance:.4f})")
        print(f"  Min / Max: {min(accs):.2f}% / {max(accs):.2f}%")

    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    for cfg, r in all_results.items():
        print(f"  {cfg}")
        print(f"    Mean +/- Std : {r['mean']:.2f} +/- {r['std']:.2f}%")
        print(f"    Variance     : {r['variance']:.4f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to results.json")

    return all_results
