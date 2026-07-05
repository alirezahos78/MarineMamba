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

## Results (seed 42 — 10-seed mean ± std pending)

### AQUA20 — 20 aquatic species

| Method | Params | Top-1 |
|--------|--------|-------|
| ResNet-50 (fine-tuned) | 25 M | ~92% |
| EfficientNet-B4 | 19 M | ~93% |
| **PyramidCLIPSpyMamba** | **1.03 M** | **94.23%** |

### Sea Animals 23 — 23 sea species

| Method | Top-1 | Top-2 | Top-3 | Macro-F1 |
|--------|-------|-------|-------|----------|
| Standard CNN (public notebook) | 88.77% | — | — | — |
| **PyramidCLIPSpyMamba** | **96.78%** | **99.23%** | **99.71%** | **96.25%** |

### Fish4Knowledge — 23 fish species (raw imbalanced split, ratio 1100:1)

| Method | Split | Top-1 | Macro-F1 |
|--------|-------|-------|----------|
| Multi-level ResVGGNet [2021] | balanced | 99.69% | — |
| **PyramidCLIPSpyMamba** | **imbalanced** | **98.46%** | **88.37%** |

> The 99.69% ResVGGNet result uses a curated balanced split; our split preserves the raw 1100:1 class imbalance, making it a strictly harder evaluation.

---

## Project Layout

```
SpyMamba/
├── main.py                         # entry point — train across multiple seeds
├── requirements.txt
├── results.json                    # populated by main.py after training
├── paper_guideline.txt             # paper writing reference
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
│   ├── infer.py                    # per-sample softmax / logit inspection
│   └── umap_vis.py                 # 6-panel UMAP visualisation (AQUA20)
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

# 3. Train across 10 seeds and report mean +/- std
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
# Use specific seeds instead of the default 0-9
python3 main.py --configs aqua20_pyramid_hybrid_128_focal_balanced --seeds 0 1 2 3 4
```

Output format after all seeds finish:
```
  aqua20_pyramid_hybrid_128_focal_balanced
    Mean +/- Std : 94.31 +/- 0.41%
    Variance     : 0.1681
```

Results are also written to `results.json`.

---

## Evaluate a Checkpoint

```bash
# Confusion matrix + per-class F1 (terminal output)
python3 scripts/confusion_matrix.py --config aqua20_pyramid_hybrid_128_focal_balanced --seed 0

# Save a PNG heatmap to logs/
python3 scripts/confusion_matrix.py --config aqua20_pyramid_hybrid_128_focal_balanced --seed 0 --save-png
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
