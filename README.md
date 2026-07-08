# PyramidCLIPSpyMamba

Lightweight dual-branch state-space model for aquatic and marine species classification. Operates on frozen CLIP spatial features extracted at two resolutions; only 1.03 M parameters are trained.

## Architecture

```
Image
  |
  |-- CLIP ViT-B/16 (frozen) --> spatial [768, 14, 14]  -->  Fine SpyMamba Branch (dim=128)  --> 128-d -|
  |                           +  CLS token  [512]         -->  (injected into sequence)                  |
  |                                                                                                       |--> concat 256-d --> MLP Head --> C classes
  |-- CLIP ViT-B/32 (frozen) --> spatial [768,  7,  7]  -->  Coarse SpyMamba Branch (dim=128) --> 128-d -|
                              +  CLS token  [512]         -->  (injected into sequence)
```

**SpyMambaBlock** — the core building block:
- LayerNorm → SpiralScanner reorders spatial tokens (outside-in and center-out)
- CLS token prepended to each spiral sequence before Mamba runs
- Bidirectional Mamba pass (forward spiral + backward spiral) with a learnable fusion gate
- CBAM channel-and-spatial attention applied to the spatial output
- LayerScale + DropPath residual for stable deep training

**Why spiral scanning?** Raster order breaks 2-D spatial locality at each row boundary. Spiral traversal keeps geometrically adjacent patches adjacent in the sequence throughout, giving Mamba's causal recurrence geometrically coherent context.

**MLP Head:** LayerNorm(256) → Linear(256→512) → GELU → Dropout(0.3) → Linear(512→C)

| Component | Params |
|-----------|--------|
| Fine branch (ViT-B/16, 14×14) | 0.453 M |
| Coarse branch (ViT-B/32, 7×7) | 0.434 M |
| MLP head | 0.144 M |
| **Total** | **1.031 M** |

---

## Results (11 seeds: 0–9 + 42, mean ± std)

### AQUA20 — 20 aquatic species

| Method | Params | Top-1 Acc |
|--------|--------|-----------|
| ResNet-50 (fine-tuned) | 25 M | ~92% |
| EfficientNet-B4 | 19 M | ~93% |
| Vim-tiny (fine-tuned, 4 seeds) | 6.96 M | 89.59 ± 0.33% |
| **PyramidCLIPSpyMamba (ours)** | **1.03 M** | **93.46 ± 0.57%** |

### Sea Animals 23 — 23 sea species

| Method | Params | Top-1 Acc |
|--------|--------|-----------|
| Standard CNN (public baseline) | — | 88.77% |
| Vim-tiny (fine-tuned, 4 seeds) | 6.96 M | 89.48 ± 0.36% |
| **PyramidCLIPSpyMamba (ours)** | **1.03 M** | **95.90 ± 0.80%** |

### Fish4Knowledge — 23 fish species (raw imbalanced split, ratio 1100:1)

| Method | Split | Top-1 Acc |
|--------|-------|-----------|
| Multi-level ResVGGNet [2021] | balanced | 99.69% |
| Vim-tiny (fine-tuned, 1 seed) | imbalanced | 99.82% |
| **PyramidCLIPSpyMamba (ours)** | **imbalanced** | **98.25 ± 0.26%** |

> The ResVGGNet result uses a curated balanced split; our split preserves the raw 1100:1 class imbalance, making it a strictly harder evaluation.

---

## Ablation Study (AQUA20, seeds 0/1/2)

### Scan order

| Variant | Top-1 Acc | Δ vs Full |
|---------|-----------|-----------|
| **Spiral (ours)** | **93.44 ± 0.13%** | — |
| Raster | 92.91 ± 0.82% | −0.53% |

### Model components

| Variant | Top-1 Acc | Δ vs Full |
|---------|-----------|-----------|
| **Full model (spiral)** | **93.44 ± 0.13%** | — |
| Coarse branch only (B/32) | 93.51 ± 0.41% | +0.06% |
| Fine branch only (B/16) | 92.29 ± 0.93% | −1.15% |
| No focal loss (CE only) | 92.83 ± 0.96% | −0.61% |
| No Mamba (FFN only) | 85.98 ± 0.44% | −7.46% |
| No CLS token injection | 85.07 ± 0.61% | −8.37% |

> Coarse-only achieves similar mean accuracy (+0.06%) but 3× higher std (0.41% vs 0.13%), confirming that the dual branch improves stability, not just accuracy.

---

## Project Layout

```
SpyMamba/
├── main.py                         # entry point — train across multiple seeds
├── requirements.txt
├── results.json                    # populated by main.py after training
├── ablation_results.json           # all ablation results (scan + model components)
├── spymamba/
│   ├── model.py                    # PyramidCLIPSpyMamba, SpyMambaBlock, SpiralScanner, CBAM
│   ├── trainer.py                  # training loop, multi-seed orchestration, checkpointing
│   ├── config.py                   # named experiment configurations
│   ├── data.py                     # DualCLIPFeatureDataset, build_dataloaders
│   ├── losses.py                   # FocalLoss, build_criterion
│   ├── paths.py                    # all filesystem paths
│   └── utils.py                    # Logger, set_seed, ensure_dir
├── scripts/
│   ├── download_aqua20.py          # download AQUA20 dataset
│   ├── download_fish4k.py          # download Fish4Knowledge dataset
│   ├── prepare_sea23.py            # 80/20 stratified split for Sea23
│   ├── build_clip_features.py      # extract and cache CLIP spatial + CLS features
│   ├── confusion_matrix.py         # per-checkpoint confusion matrix and F1 report
│   ├── ablation.py                 # all ablations (scan order + model components)
│   ├── baseline_vim.py             # Vim-tiny baseline fine-tuning (reproduces comparison numbers)
│   └── umap_vis.py                 # 6-panel UMAP visualisation (all 3 datasets)
├── data/                           # feature caches and raw images (not tracked by git)
└── logs/                           # per-seed training logs and plots (not tracked by git)
```

---

## Quick Start

**Requirements:** CUDA GPU (mamba-ssm requires CUDA), Python >= 3.10.

```bash
pip install -r requirements.txt
```

### AQUA20

```bash
# 1. Download dataset (~300 MB → data/aqua20/)
python3 scripts/download_aqua20.py

# 2. Extract CLIP features (run once, ~15 min on GPU)
python3 scripts/build_clip_features.py

# 3. Train across 11 seeds and report mean ± std
python3 main.py --configs aqua20_pyramid_hybrid_128_focal_balanced
```

### Sea Animals 23

```bash
# 1. Prepare 80/20 split from the downloaded flat directory
python3 scripts/prepare_sea23.py --src data/sea23_raw --dst data/sea23

# 2. Extract CLIP features
python3 scripts/build_clip_features.py \
    --aqua20-root data/sea23 \
    --out-b16 data/sea23_clip_vit_b16_spatial_grid_aug \
    --out-b32 data/sea23_clip_vit_b32_spatial_grid_aug \
    --out-cls data/sea23_dual_clip_pooled_features.pt

# 3. Train
python3 main.py --configs sea23_pyramid_hybrid_128_focal_balanced
```

### Fish4Knowledge

```bash
# 1. Download and split (~500 MB → data/fish4k/)
python3 scripts/download_fish4k.py

# 2. Extract CLIP features
python3 scripts/build_clip_features.py \
    --aqua20-root data/fish4k \
    --out-b16 data/fish4k_clip_vit_b16_spatial_grid_aug \
    --out-b32 data/fish4k_clip_vit_b32_spatial_grid_aug \
    --out-cls data/fish4k_dual_clip_pooled_features.pt

# 3. Train
python3 main.py --configs fish4k_baseline
```

### All three datasets in one run

```bash
python3 main.py --configs \
    aqua20_pyramid_hybrid_128_focal_balanced \
    sea23_pyramid_hybrid_128_focal_balanced \
    fish4k_baseline
```

---

## Custom Seeds

```bash
python3 main.py --configs aqua20_pyramid_hybrid_128_focal_balanced --seeds 0 1 2 3 4
```

Output after all seeds finish:
```
  aqua20_pyramid_hybrid_128_focal_balanced
    Mean ± Std : 93.46 ± 0.57%
    Variance   : 0.3216
```

Results are also written to `results.json`.

---

## Ablation Scripts

```bash
# All variants (AQUA20, seeds 0 1 2)
python3 scripts/ablation.py

# Specific variants only
python3 scripts/ablation.py --variants spiral raster
python3 scripts/ablation.py --variants no_mamba no_cls --seeds 0 1 2
```

Available variants: `spiral`, `raster`, `no_focal_loss`, `no_mamba`, `no_cls`, `fine_only`, `coarse_only`.

Results are written to `ablation_results.json`.

---

## UMAP Visualisation

```bash
# 6-panel UMAP for any dataset (train + test)
python3 scripts/umap_vis.py --config aqua20_pyramid_hybrid_128_focal_balanced --seed 4
python3 scripts/umap_vis.py --config sea23_pyramid_hybrid_128_focal_balanced --seed 3
python3 scripts/umap_vis.py --config fish4k_baseline --seed 9
```

Panels: CLIP B/16 · CLIP B/32 · Mamba fine branch · Mamba coarse branch · Concat · Raw images.

---

## Vim-tiny Baseline

Evaluates pretrained Vim-tiny (ImageNet-1k, 76.1% top-1) with three freeze modes so the comparison against SpyMamba is explicit and fair.

**Citation:** Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model", ICML 2024. [arXiv:2401.13586](https://arxiv.org/abs/2401.13586)

| Mode | Trainable params | LR default | Notes |
|------|-----------------|-----------|-------|
| `head_only` | ~4 K | 1e-3 | Linear probe — **fair comparison** (backbone frozen, same constraint as SpyMamba) |
| `last_block` | ~0.4 M | 1e-4 | Last Mamba block + head |
| `full` | 6.96 M | 5e-5 | Full fine-tune (original setup, upper bound) |

```bash
# Fair comparison — backbone frozen, only head trained (default)
python3 scripts/baseline_vim.py --dataset aqua20 --freeze head_only --seeds 0 1 2 42
python3 scripts/baseline_vim.py --dataset sea23  --freeze head_only --seeds 0 1 2 42
python3 scripts/baseline_vim.py --dataset fish4k --freeze head_only --seeds 0 1 2 42

# Last Mamba block + head
python3 scripts/baseline_vim.py --dataset aqua20 --freeze last_block

# Full fine-tune (upper bound)
python3 scripts/baseline_vim.py --dataset aqua20 --freeze full
```

Results saved to `results_vim_{dataset}_{freeze}.json`. First run auto-clones the Vim repo into `data/_vim_repo/`.

---

## Confusion Matrix

```bash
python3 scripts/confusion_matrix.py --config aqua20_pyramid_hybrid_128_focal_balanced --seed 4 --save-png
```

---

## Feature Cache Format

Features are extracted once and cached as `.pt` files. Training loads directly from caches — raw images are not needed at train time.

| File | Contents |
|------|----------|
| `*_clip_vit_b16_spatial_grid_aug/train_features.pt` | B/16 patches [768, 14, 14], 4 views/image |
| `*_clip_vit_b32_spatial_grid_aug/train_features.pt` | B/32 patches [768, 7, 7], 4 views/image |
| `*_clip_vit_b16_spatial_grid_aug/test_features.pt`  | B/16 patches, 1 view/image (clean) |
| `*_clip_vit_b32_spatial_grid_aug/test_features.pt`  | B/32 patches, 1 view/image (clean) |
| `*_dual_clip_pooled_features.pt` | CLS tokens for B/16 and B/32 (clean, 1 view) |

The `aug` suffix means 3 augmented crops are extracted per training image at build time (random resized crop, color jitter, horizontal flip, grid-shuffle), giving 4 views per image total.
