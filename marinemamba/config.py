import os
from copy import deepcopy
from .paths import (B16_FEATURES_ROOT, B32_FEATURES_ROOT, CLS_FEATURES_PATH,
                    SEA23_B16_ROOT, SEA23_B32_ROOT, SEA23_CLS_PATH,
                    FISH4K_B16_ROOT, FISH4K_B32_ROOT, FISH4K_CLS_PATH)


def _base():
    return {
        "num_classes": 20,
        "fine_train_path": os.path.join(B16_FEATURES_ROOT, "train_features.pt"),
        "fine_test_path":  os.path.join(B16_FEATURES_ROOT, "test_features.pt"),
        "fine_input_dim": 768, "fine_h": 14, "fine_w": 14, "fine_depth": 1,
        "coarse_train_path": os.path.join(B32_FEATURES_ROOT, "train_features.pt"),
        "coarse_test_path":  os.path.join(B32_FEATURES_ROOT, "test_features.pt"),
        "coarse_input_dim": 768, "coarse_h": 7, "coarse_w": 7, "coarse_depth": 1,
        "dim": 128, "cls_dim": 512, "head_hidden_dim": 512,
        "stochastic_depth": 0.0, "ffn_drop": 0.0,
        "layer_scale_init": 1.0, "dropout": 0.3,
        "cls_path": CLS_FEATURES_PATH,
        "batch_size": 128, "epochs": 60, "lr": 3e-4, "weight_decay": 0.05,
        "label_smoothing": 0.0, "warmup_epochs": 5,
        "balanced_sampling": "sqrt", "early_stopping_patience": 15,
        "num_workers": 2,
        "loss": "focal",
        "focal_gamma": 2.0,
        "focal_alpha": "balanced",
    }


CONFIGS = {
    # ── AQUA20 (20 classes) ──────────────────────────────────────────────────
    "aqua20_dual_hybrid_128_focal_balanced": {
        **_base(),
        "name": "AQUA20 Hybrid dim=128 + Focal (gamma=2, balanced alpha)",
    },

    # ── Sea Animals 23 (23 classes) ──────────────────────────────────────────
    "sea23_dual_hybrid_128_focal_balanced": {
        **_base(),
        "name": "SEA23 Hybrid dim=128 + Focal (gamma=2, balanced alpha)",
        "num_classes": 23,
        "fine_train_path":   os.path.join(SEA23_B16_ROOT, "train_features.pt"),
        "fine_test_path":    os.path.join(SEA23_B16_ROOT, "test_features.pt"),
        "coarse_train_path": os.path.join(SEA23_B32_ROOT, "train_features.pt"),
        "coarse_test_path":  os.path.join(SEA23_B32_ROOT, "test_features.pt"),
        "cls_path": SEA23_CLS_PATH,
    },

    # ── Fish4Knowledge 23 (23 classes) ───────────────────────────────────────
    "fish4k_dual_hybrid_128_focal_balanced": {
        **_base(),
        "name": "Fish4K Focal + sqrt sampler (baseline)",
        "num_classes": 23,
        "fine_train_path":   os.path.join(FISH4K_B16_ROOT, "train_features.pt"),
        "fine_test_path":    os.path.join(FISH4K_B16_ROOT, "test_features.pt"),
        "coarse_train_path": os.path.join(FISH4K_B32_ROOT, "train_features.pt"),
        "coarse_test_path":  os.path.join(FISH4K_B32_ROOT, "test_features.pt"),
        "cls_path": FISH4K_CLS_PATH,
    },

}

RUN_SEED = 42


def get_config(name):
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS)}")
    return deepcopy(CONFIGS[name])
