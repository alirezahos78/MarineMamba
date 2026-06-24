import gc
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .augmentations import cutmix_data, mixup_criterion, mixup_data
from .config import DEFAULT_DATASETS_TO_RUN, RUN_SEED, get_dataset_config
from .data import build_dataloaders
from .paths import LOGS_DIR, PROJECT_ROOT, RESULTS_PATH
from .utils import Logger, ensure_dir, set_seed


def get_patch_size(img_size):
    if img_size >= 64:
        return 4
    if img_size >= 32:
        return 2
    return 1


def build_model(config, branch_setting, device):
    from .model import UniversalMambaTinyImageNet, get_branch_flags

    branch_flags = get_branch_flags(branch_setting)
    model = UniversalMambaTinyImageNet(
        patch_size=config.get("patch_size", get_patch_size(config["img_size"])),
        dim=128,
        depth=6,
        num_classes=config["num_classes"],
        img_size=config["img_size"],
        in_channels=config["channels"],
        stochastic_depth=config.get("stochastic_depth", 0.0),
        branch_flags=branch_flags,
    ).to(device)
    return model


def build_feature_classifier(config, device):
    return nn.Sequential(
        nn.LayerNorm(config["feature_dim"]),
        nn.Linear(config["feature_dim"], config.get("hidden_dim", 512)),
        nn.GELU(),
        nn.Dropout(config.get("dropout", 0.0)),
        nn.Linear(config.get("hidden_dim", 512), config["num_classes"]),
    ).to(device)


def build_spatial_feature_mamba(config, branch_setting, device):
    from .model import SpatialFeatureMambaClassifier, get_branch_flags

    branch_flags = get_branch_flags(branch_setting)
    return SpatialFeatureMambaClassifier(
        input_dim=config["feature_dim"],
        feature_height=config["feature_height"],
        feature_width=config["feature_width"],
        patch_size=config.get("patch_size", 1),
        dim=config.get("model_dim", 128),
        depth=config.get("depth", 6),
        num_classes=config["num_classes"],
        stochastic_depth=config.get("stochastic_depth", 0.0),
        branch_flags=branch_flags,
        dropout=config.get("dropout", 0.15),
    ).to(device)


def load_pretrained_weights(model, checkpoint_path, device):
    if not checkpoint_path:
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped = sorted(set(state_dict) - set(compatible))
    model_state.update(compatible)
    model.load_state_dict(model_state)
    print(f"🔁 Loaded {len(compatible)} tensors from {checkpoint_path}")
    if skipped:
        print(f"   Skipped {len(skipped)} incompatible tensors, usually the classifier head.")


def freeze_first_blocks_for_transfer(model, num_blocks):
    if not num_blocks:
        return 0
    if num_blocks < 0:
        raise ValueError("freeze_first_blocks_on_transfer must be >= 0")
    if num_blocks >= len(model.blocks):
        raise ValueError(
            f"freeze_first_blocks_on_transfer={num_blocks} would freeze all {len(model.blocks)} blocks"
        )

    for block in model.blocks[:num_blocks]:
        for parameter in block.parameters():
            parameter.requires_grad = False

    frozen_params = sum(
        parameter.numel()
        for block in model.blocks[:num_blocks]
        for parameter in block.parameters()
    )
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"🧊 Frozen first {num_blocks} Mamba blocks for transfer learning; "
        f"{trainable_params / 1e6:.2f}M trainable params remain."
    )
    return frozen_params


def set_frozen_blocks_eval(model, num_blocks):
    if not num_blocks:
        return
    for block in model.blocks[:num_blocks]:
        block.eval()


def make_run_name(dataset_name, branch_setting, seed=None):
    run_name = f"{dataset_name}_spiral_{branch_setting}"
    if seed is not None:
        run_name = f"{run_name}_seed_{seed}"
    return run_name


def is_cuda_device(device):
    return torch.device(device).type == "cuda"


def move_batch_to_device(inputs, targets, device):
    non_blocking = is_cuda_device(device)
    return inputs.to(device, non_blocking=non_blocking), targets.to(device, non_blocking=non_blocking)


def release_cuda_memory(device, empty_cache=False):
    gc.collect()
    if empty_cache and is_cuda_device(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def build_criterion(config, trainset=None, label_smoothing=None, device="cpu"):
    label_smoothing = config["label_smoothing"] if label_smoothing is None else label_smoothing
    class_weighting = config.get("class_weighting")
    if not class_weighting or trainset is None:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    targets = torch.as_tensor(trainset.targets, dtype=torch.long)
    class_counts = torch.bincount(targets, minlength=config["num_classes"]).float()
    if torch.any(class_counts == 0):
        raise ValueError("Class-weighted loss requires at least one training sample per class")

    if class_weighting == "sqrt":
        class_weights = class_counts.rsqrt()
    elif class_weighting in {"inverse", "uniform"}:
        class_weights = class_counts.reciprocal()
    else:
        raise ValueError(f"Unknown class_weighting mode: {class_weighting}")

    class_weights = class_weights * (config["num_classes"] / class_weights.sum())
    print(
        f"⚖️ Using {class_weighting} class-weighted softmax loss "
        f"(min={class_weights.min().item():.3f}, max={class_weights.max().item():.3f})"
    )
    return nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing,
    )


def train_on_dataset(dataset_name, config, device, branch_setting="full", seed=42):
    run_name = make_run_name(dataset_name, branch_setting, seed)

    print("\n" + "=" * 60)
    print(f"🚀 Training on {config['name']}")
    print(f"   Mode: {'CLIP features' if config.get('feature_dataset') else 'spiral image model'}")
    if not config.get("feature_dataset"):
        print("   Scan Type: spiral")
        print(f"   Branch Setting: {branch_setting}")
    print(f"   Seed: {seed}")
    if config.get("feature_dataset"):
        if config.get("spatial_feature_dataset"):
            print(f"   Feature Grid: {config['feature_dim']}x{config['feature_height']}x{config['feature_width']}")
            print(f"   Patch Size: {config.get('patch_size', 1)}x{config.get('patch_size', 1)}")
            print(f"   Mamba Dim: {config.get('model_dim', 128)}")
            print(f"   Branch Setting: {branch_setting}")
        else:
            print(f"   Feature Dim: {config['feature_dim']}")
    else:
        print(f"   Image Size: {config['img_size']}x{config['img_size']}")
        print(f"   Patch Size: {config.get('patch_size', get_patch_size(config['img_size']))}x{config.get('patch_size', get_patch_size(config['img_size']))}")
        print("   Local Downsample Scales: 2x, 4x, 8x")
    print(f"   Classes: {config['num_classes']}")
    if not config.get("feature_dataset"):
        print(f"   Channels: {config['channels']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch Size: {config['batch_size']}")
    print("=" * 60)

    ensure_dir(LOGS_DIR)
    original_stdout = sys.stdout
    logger = Logger(os.path.join(LOGS_DIR, f"log_{run_name}.txt"))
    sys.stdout = logger
    try:
        return _train_on_dataset_logged(dataset_name, config, device, branch_setting, run_name, seed)
    finally:
        sys.stdout = original_stdout
        logger.close()


def _train_on_dataset_logged(dataset_name, config, device, branch_setting, run_name, seed):
    if config.get("spatial_feature_dataset"):
        return _train_on_spatial_feature_dataset_logged(dataset_name, config, device, branch_setting, run_name, seed)
    if config.get("feature_dataset"):
        return _train_on_feature_dataset_logged(dataset_name, config, device, branch_setting, run_name, seed)

    set_seed(seed)
    model = build_model(config, branch_setting, device)
    load_pretrained_weights(model, config.get("pretrained_path"), device)
    frozen_blocks = 0
    if config.get("pretrained_path"):
        frozen_blocks = config.get("freeze_first_blocks_on_transfer", 0)
        freeze_first_blocks_for_transfer(model, frozen_blocks)
    from .metrics import compute_flops, compute_latency, compute_memory

    input_size = (1, config["channels"], config["img_size"], config["img_size"])
    flops_count, params_count = compute_flops(model, input_size, device)
    latency_val = compute_latency(model, input_size, device)
    memory_val = compute_memory(model, input_size, device)

    print(f"\n📊 Model Statistics for {config['name']}:")
    print(f"   Parameters: {params_count / 1e6:.2f} M")
    print(f"   FLOPs: {flops_count / 1e9:.2f} G")
    print(f"   Latency: {latency_val:.2f} ms")
    print(f"   Memory: {memory_val:.2f} MB")

    print(f"\n📥 Loading {config['name']} dataset...")
    trainloader, testloader = build_dataloaders(dataset_name, config)

    criterion = build_criterion(config, trainloader.dataset, device=device)
    test_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    total_steps = config["epochs"] * len(trainloader)
    warmup_steps = config["warmup_epochs"] * len(trainloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        t_max = total_steps - warmup_steps
        t_cur = step - warmup_steps
        return 0.5 * (1.0 + math.cos(math.pi * t_cur / t_max))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_acc = 0.0
    epochs_without_improvement = 0
    early_stopping_patience = config.get("early_stopping_patience")

    for epoch in range(config["epochs"]):
        model.train()
        set_frozen_blocks_eval(model, frozen_blocks)
        train_loss = 0.0
        correct_train = 0.0
        total_train = 0
        epoch_lambdas = {f"block_{idx}": {} for idx in range(6)}

        loop = tqdm(trainloader, desc=f"Ep {epoch + 1}/{config['epochs']}", leave=False)
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = move_batch_to_device(inputs, targets, device)
            optimizer.zero_grad(set_to_none=True)

            use_mixing = torch.rand(1).item() < config.get("mix_probability", 1.0)
            if use_mixing and torch.rand(1).item() < 0.5:
                inputs_mixed, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, config["mixup_alpha"], device
                )
            elif use_mixing:
                inputs_mixed, targets_a, targets_b, lam = cutmix_data(
                    inputs, targets, config["cutmix_alpha"], device
                )
            else:
                inputs_mixed = inputs
                targets_a = targets_b = targets
                lam = 1.0

            student_features = model.forward_features(inputs_mixed)
            outputs = model.head(student_features)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            _, predicted = outputs.max(1)
            correct_train += (predicted.eq(targets_a) * lam + predicted.eq(targets_b) * (1 - lam)).sum().item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_value = loss.item()
            train_loss += loss_value
            total_train += targets.size(0)
            loop.set_postfix(loss=loss_value)

            if batch_idx == len(trainloader) - 1:
                with torch.inference_mode():
                    lambda_outputs, all_lambdas = model(inputs[:1], return_lambdas=True)
                    for block_idx, lambdas in enumerate(all_lambdas):
                        epoch_lambdas[f"block_{block_idx}"].update(lambdas)
                del lambda_outputs, all_lambdas

            del inputs, targets, inputs_mixed, targets_a, targets_b, outputs, loss, predicted
            del student_features

        release_cuda_memory(device)
        train_acc = 100.0 * correct_train / total_train
        avg_train_loss = train_loss / len(trainloader)

        print(f"\n{'=' * 70}")
        print(f"📊 Lambda Values at Epoch {epoch + 1}:")
        print(f"{'=' * 70}")
        for block_name, lambdas in epoch_lambdas.items():
            if lambdas:
                lambda_str = " | ".join(f"{key}: {value:.4f}" for key, value in lambdas.items())
                print(f"  {block_name}: {lambda_str}")
        print(f"{'=' * 70}\n")

        model.eval()
        correct_test = 0
        total_test = 0

        with torch.inference_mode():
            for inputs, targets in testloader:
                inputs, targets = move_batch_to_device(inputs, targets, device)
                outputs = model(inputs)
                _ = test_criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                total_test += targets.size(0)
                correct_test += predicted.eq(targets).sum().item()
                del inputs, targets, outputs, predicted

        test_acc = 100.0 * correct_test / total_test
        if test_acc > best_acc:
            best_acc = test_acc
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(PROJECT_ROOT, f"best_model_{run_name}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved best model for {config['name']} ({branch_setting}, seed {seed}): {best_acc:.2f}%")
        else:
            epochs_without_improvement += 1

        print(
            f"{config['name']} [spiral, {branch_setting}, seed {seed}] | Ep {epoch + 1:02d} | "
            f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_acc:.2f}% | "
            f"Loss: {avg_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        if config.get("empty_cache_each_epoch", False):
            release_cuda_memory(device, empty_cache=True)
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            print(
                f"🛑 Early stopping after {early_stopping_patience} epochs without improvement. "
                f"Best test accuracy: {best_acc:.2f}%"
            )
            break

    return {
        "name": config["name"],
        "scan_type": "spiral",
        "branch_setting": branch_setting,
        "seed": seed,
        "accuracy": best_acc,
        "latency_ms": latency_val,
        "memory_mb": memory_val,
        "params_m": params_count / 1e6,
        "img_size": config["img_size"],
        "num_classes": config["num_classes"],
    }


def _train_on_spatial_feature_dataset_logged(dataset_name, config, device, branch_setting, run_name, seed):
    set_seed(seed)
    model = build_spatial_feature_mamba(config, branch_setting, device)

    params_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"\n📊 Spatial Feature SpyMamba Statistics for {config['name']}:")
    print(f"   Parameters: {params_count / 1e6:.2f} M")
    print(f"   Input Grid: {config['feature_dim']}x{config['feature_height']}x{config['feature_width']}")
    print(f"   Patch Size: {config.get('patch_size', 1)}x{config.get('patch_size', 1)}")
    print(
        f"   Tokens: "
        f"{(config['feature_height'] // config.get('patch_size', 1)) * (config['feature_width'] // config.get('patch_size', 1))}"
    )
    print(f"   Model Dropout: {config.get('dropout', 0.15)}")
    print(f"   Stochastic Depth: {config.get('stochastic_depth', 0.0)}")

    print(f"\n📥 Loading {config['name']} dataset...")
    trainloader, testloader = build_dataloaders(dataset_name, config)

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    test_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    total_steps = config["epochs"] * len(trainloader)
    warmup_steps = config["warmup_epochs"] * len(trainloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        t_max = total_steps - warmup_steps
        t_cur = step - warmup_steps
        return 0.5 * (1.0 + math.cos(math.pi * t_cur / t_max))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_acc = 0.0
    epochs_without_improvement = 0
    early_stopping_patience = config.get("early_stopping_patience")

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_lambdas = {f"block_{idx}": {} for idx in range(config.get("depth", 6))}

        loop = tqdm(trainloader, desc=f"Ep {epoch + 1}/{config['epochs']}", leave=False)
        for batch_idx, (features, targets) in enumerate(loop):
            features, targets = move_batch_to_device(features, targets, device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(features)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(targets).sum().item()
            total_train += targets.size(0)

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_value = loss.item()
            train_loss += loss_value
            loop.set_postfix(loss=loss_value)

            if batch_idx == len(trainloader) - 1:
                with torch.inference_mode():
                    lambda_outputs, all_lambdas = model(features[:1], return_lambdas=True)
                    for block_idx, lambdas in enumerate(all_lambdas):
                        epoch_lambdas[f"block_{block_idx}"].update(lambdas)
                del lambda_outputs, all_lambdas

            del features, targets, outputs, loss, predicted

        train_acc = 100.0 * correct_train / total_train
        avg_train_loss = train_loss / len(trainloader)

        print(f"\n{'=' * 70}")
        print(f"📊 Lambda Values at Epoch {epoch + 1}:")
        print(f"{'=' * 70}")
        for block_name, lambdas in epoch_lambdas.items():
            if lambdas:
                lambda_str = " | ".join(f"{key}: {value:.4f}" for key, value in lambdas.items())
                print(f"  {block_name}: {lambda_str}")
        print(f"{'=' * 70}\n")

        model.eval()
        correct_test = 0
        total_test = 0
        with torch.inference_mode():
            for features, targets in testloader:
                features, targets = move_batch_to_device(features, targets, device)
                outputs = model(features)
                _ = test_criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                total_test += targets.size(0)
                correct_test += predicted.eq(targets).sum().item()
                del features, targets, outputs, predicted

        test_acc = 100.0 * correct_test / total_test
        if test_acc > best_acc:
            best_acc = test_acc
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(PROJECT_ROOT, f"best_model_{run_name}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved best model for {config['name']} ({branch_setting}, seed {seed}): {best_acc:.2f}%")
        else:
            epochs_without_improvement += 1

        print(
            f"{config['name']} [spatial CLIP SpyMamba, {branch_setting}, seed {seed}] | Ep {epoch + 1:02d} | "
            f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_acc:.2f}% | "
            f"Loss: {avg_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            print(
                f"🛑 Early stopping after {early_stopping_patience} epochs without improvement. "
                f"Best test accuracy: {best_acc:.2f}%"
            )
            break

    return {
        "name": config["name"],
        "scan_type": "clip_spatial_spymamba",
        "branch_setting": branch_setting,
        "seed": seed,
        "accuracy": best_acc,
        "latency_ms": 0.0,
        "memory_mb": 0.0,
        "params_m": params_count / 1e6,
        "img_size": None,
        "feature_dim": config["feature_dim"],
        "feature_shape": f"{config['feature_dim']}x{config['feature_height']}x{config['feature_width']}",
        "num_classes": config["num_classes"],
    }


def _train_on_feature_dataset_logged(dataset_name, config, device, branch_setting, run_name, seed):
    set_seed(seed)
    model = build_feature_classifier(config, device)

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"\n📊 Feature Classifier Statistics for {config['name']}:")
    print(f"   Parameters: {trainable_params / 1e6:.2f} M")
    print(f"   Input Features: {config['feature_dim']}")

    print(f"\n📥 Loading {config['name']} dataset...")
    trainloader, testloader = build_dataloaders(dataset_name, config)

    criterion = build_criterion(config, trainloader.dataset, device=device)
    test_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    total_steps = config["epochs"] * len(trainloader)
    warmup_steps = config["warmup_epochs"] * len(trainloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        t_max = total_steps - warmup_steps
        t_cur = step - warmup_steps
        return 0.5 * (1.0 + math.cos(math.pi * t_cur / t_max))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_acc = 0.0
    epochs_without_improvement = 0
    early_stopping_patience = config.get("early_stopping_patience")

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        loop = tqdm(trainloader, desc=f"Ep {epoch + 1}/{config['epochs']}", leave=False)
        for features, targets in loop:
            features, targets = move_batch_to_device(features, targets, device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(features)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(targets).sum().item()
            total_train += targets.size(0)

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_value = loss.item()
            train_loss += loss_value
            loop.set_postfix(loss=loss_value)
            del features, targets, outputs, loss, predicted

        train_acc = 100.0 * correct_train / total_train
        avg_train_loss = train_loss / len(trainloader)

        model.eval()
        correct_test = 0
        total_test = 0
        with torch.inference_mode():
            for features, targets in testloader:
                features, targets = move_batch_to_device(features, targets, device)
                outputs = model(features)
                _ = test_criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                total_test += targets.size(0)
                correct_test += predicted.eq(targets).sum().item()
                del features, targets, outputs, predicted

        test_acc = 100.0 * correct_test / total_test
        if test_acc > best_acc:
            best_acc = test_acc
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(PROJECT_ROOT, f"best_model_{run_name}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved best model for {config['name']} ({branch_setting}, seed {seed}): {best_acc:.2f}%")
        else:
            epochs_without_improvement += 1

        print(
            f"{config['name']} [CLIP features, {branch_setting}, seed {seed}] | Ep {epoch + 1:02d} | "
            f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_acc:.2f}% | "
            f"Loss: {avg_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            print(
                f"🛑 Early stopping after {early_stopping_patience} epochs without improvement. "
                f"Best test accuracy: {best_acc:.2f}%"
            )
            break

    return {
        "name": config["name"],
        "scan_type": "clip_features",
        "branch_setting": branch_setting,
        "seed": seed,
        "accuracy": best_acc,
        "latency_ms": 0.0,
        "memory_mb": 0.0,
        "params_m": trainable_params / 1e6,
        "img_size": None,
        "feature_dim": config["feature_dim"],
        "num_classes": config["num_classes"],
    }


def run_experiments(datasets_to_run=None, branch_settings=None, config_overrides=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")

    datasets_to_run = datasets_to_run or DEFAULT_DATASETS_TO_RUN
    requested_branch_settings = branch_settings
    config_overrides = config_overrides or {}
    seed = RUN_SEED
    results = {}

    for dataset_name in datasets_to_run:
        config = get_dataset_config(dataset_name)
        config.update({key: value for key, value in config_overrides.items() if value is not None})
        branch_settings = requested_branch_settings
        if branch_settings is None:
            if config.get("spatial_feature_dataset"):
                branch_settings = ["no_local_8"]
            elif config.get("feature_dataset"):
                branch_settings = ["clip"]
            else:
                branch_settings = ["full", "no_local_2"]

        for branch_setting in branch_settings:
            print(f"\n{'=' * 70}")
            print(f"🎯 Starting training on {config['name'].upper()} [SPIRAL, {branch_setting.upper()}, SEED {seed}]")
            print(f"{'=' * 70}")

            result_key = make_run_name(dataset_name, branch_setting, seed)
            try:
                results[result_key] = train_on_dataset(
                    dataset_name,
                    config,
                    device,
                    branch_setting=branch_setting,
                    seed=seed,
                )
            except Exception as exc:
                print(f"❌ Error training on {dataset_name} [spiral, {branch_setting}, seed {seed}]: {exc}")
                continue
            finally:
                release_cuda_memory(device, empty_cache=True)

    return results


def summarize_results(results):
    return [
        {
            "name": metrics["name"],
            "branch_setting": metrics["branch_setting"],
            "seed": metrics["seed"],
            "accuracy": metrics["accuracy"],
        }
        for metrics in results.values()
    ]


def print_accuracy_chart(summaries):
    if not summaries:
        return

    print("\n" + "=" * 80)
    print("📊 ACCURACY CHART")
    print("=" * 80)
    max_accuracy = max(summary["accuracy"] for summary in summaries)
    scale = 40.0 / max_accuracy if max_accuracy > 0 else 1.0
    print(f"{'Dataset':<20} | {'Seed':<6} | {'Accuracy':<9} | Chart")
    print("-" * 82)
    for summary in summaries:
        label = summary["name"]
        if summary["branch_setting"] != "full":
            label = f"{label} ({summary['branch_setting']})"
        bar = "#" * max(1, int(round(summary["accuracy"] * scale)))
        print(
            f"{label:<20} | {summary['seed']:<6} | {summary['accuracy']:.2f}%{'':<3} | {bar}"
        )
    print("=" * 80)


def save_and_print_results(results):
    if not results:
        print("\n❌ No successful training completed.")
        return

    print("\n" + "=" * 80)
    print("🏆 FINAL RESULTS")
    print("=" * 80)
    print(
        f"{'Dataset':<20} | {'Seed':<6} | {'Scan':<10} | {'Branches':<12} | {'Img Size':<10} | "
        f"{'Classes':<8} | {'Accuracy':<12} | {'Params (M)':<12} | {'Latency (ms)':<12} | {'Memory (MB)':<12}"
    )
    print("-" * 147)
    for metrics in results.values():
        input_shape = (
            metrics["feature_shape"]
            if metrics.get("feature_shape")
            else f"{metrics['feature_dim']} feat"
            if metrics.get("feature_dim")
            else f"{metrics['img_size']}x{metrics['img_size']}"
        )
        print(
            f"{metrics['name']:<20} | {metrics['seed']:<6} | {metrics['scan_type']:<10} | {metrics['branch_setting']:<12} | "
            f"{input_shape:<10} | {metrics['num_classes']:<8} | "
            f"{metrics['accuracy']:.2f}%{'':<6} | {metrics['params_m']:.2f}{'':<10} | "
            f"{metrics['latency_ms']:.2f}{'':<10} | {metrics['memory_mb']:.2f}"
        )
    print("=" * 80)

    summaries = summarize_results(results)
    print_accuracy_chart(summaries)

    results_payload = {
        "seed": RUN_SEED,
        "runs": results,
        "summary": summaries,
    }
    with open(RESULTS_PATH, "w") as handle:
        json.dump(results_payload, handle, indent=4)
    print("\n💾 Results saved to spiral_results_all_datasets.json")
