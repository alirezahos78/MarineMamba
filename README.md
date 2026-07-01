# SpyMamba

Underwater species classification using a dual-branch Mamba model over pre-extracted CLIP spatial features, with a CLIP CLS token injected directly into each Mamba sequence for global context.

## Architecture — PyramidCLIPSpyMamba

```
Image
  │
  ├─ CLIP ViT-B/16 ──► spatial [768, 14, 14] ──► SpyMamba branch (dim=128) ──► 128-d ─┐
  │                                                                                       ├─ concat
  └─ CLIP ViT-B/32 ──► spatial [768,  7,  7] ──► SpyMamba branch (dim=128) ──► 128-d ─┘
                                                                                         │
                                                                                    256-d
                                                                                         │
                                                                              MLP head (LayerNorm → 512 → GELU → Dropout → C)
```

Each branch prepends the corresponding CLIP CLS token to the spatial patch sequence before the Mamba blocks, giving every token access to global semantic context. The bidirectional spiral scan orders patches from the outside in (and back), and CBAM spatial+channel attention is applied after each Mamba pass.

Key design choices:
- **Spiral bidirectional scan** — processes patches outside-in for efficient long-range coverage
- **CLS-in-sequence** — global context injected as a learnable position in the Mamba sequence, not as a skip connection
- **Dual-scale pyramid** — B/16 (fine, 14×14) and B/32 (coarse, 7×7) branches trained jointly
- **Focal loss + balanced alpha** — inverse-frequency class weights, γ=2
- **1.03M parameters** total

## Results

### AQUA20 (20 underwater species classes)

| Model | Top-1 | Top-2 | Top-3 | Macro-F1 |
|---|---|---|---|---|
| CLIP ViT-B/16 CLS baseline | 88.83% | — | — | — |
| CLIP ViT-B/32 CLS baseline | 86.54% | — | — | — |
| CLIP Dual CLS (B/16 + B/32) | 89.76% | — | — | — |
| **PyramidCLIPSpyMamba (ours)** | **93.67%** | **98.01%** | **98.88%** | **84.9%** |

### Sea Animals 23 (23 class Kaggle dataset)

| Model | Top-1 |
|---|---|
| **PyramidCLIPSpyMamba (ours)** | **94.95%** |

## Project Layout

```
SpyMamba/
├── main.py                       # entry point — train one or more configs
├── requirements.txt
├── spymamba/                     # core library
│   ├── model.py                  # PyramidCLIPSpyMamba, SpyMambaBlock, SpiralScanner
│   ├── trainer.py                # training loop, evaluation, checkpointing
│   ├── config.py                 # named experiment configs
│   ├── data.py                   # DualCLIPFeatureDataset, dataloaders
│   ├── losses.py                 # FocalLoss, build_criterion
│   ├── paths.py                  # all filesystem paths (self-contained)
│   └── utils.py                  # Logger, set_seed, ensure_dir
├── scripts/
│   ├── download_aqua20.py        # step 1 — download AQUA20 dataset
│   ├── prepare_sea23.py          # step 1 (Sea23) — 80/20 split of Kaggle flat download
│   ├── build_clip_features.py    # step 2 — extract CLIP spatial + CLS features
│   └── confusion_matrix.py       # evaluate a trained checkpoint
├── data/                         # feature caches and raw images (not tracked by git)
└── logs/                         # training logs and plots (not tracked by git)
```

## Quick Start

> **Requirements:** CUDA GPU (mamba-ssm requires CUDA), Python ≥ 3.10.

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Download AQUA20 (~1 GB → data/aqua20/)
python3 scripts/download_aqua20.py

# 2. Extract CLIP features (~15 min on GPU, saved to data/)
python3 scripts/build_clip_features.py

# 3. Train
python3 main.py
```

The model trains on AQUA20 by default and saves the best checkpoint to
`best_model_aqua20_pyramid_hybrid_128_focal_balanced_seed_42.pth`.

## Sea Animals 23

```bash
# Download from Kaggle (requires ~/.kaggle/kaggle.json)
kaggle datasets download -d vencerlanz09/sea-animals-image-dataste -p data/sea23_raw --unzip

# Prepare 80/20 train/test split
python3 scripts/prepare_sea23.py --src data/sea23_raw

# Extract CLIP features for Sea23
python3 scripts/build_clip_features.py \
    --dataset sea23 \
    --aqua20-root data/sea23 \
    --out-b16 data/sea23_clip_vit_b16_spatial_grid_aug \
    --out-b32 data/sea23_clip_vit_b32_spatial_grid_aug \
    --out-cls data/sea23_dual_clip_pooled_features.pt

# Train
python3 main.py --configs sea23_pyramid_hybrid_128_focal_balanced
```

## Evaluate — Confusion Matrix

```bash
python3 scripts/confusion_matrix.py --save-png
# output: logs/confusion_matrix_aqua20_pyramid_hybrid_128_focal_balanced.png
```

## Feature Caches

`scripts/build_clip_features.py` extracts features once and caches them as `.pt` files.
Training loads directly from these caches — no images needed at train time.

| Cache file | Contents |
|---|---|
| `data/aqua20_clip_vit_b16_spatial_grid_aug/train_features.pt` | B/16 spatial [768,14,14], 4 views/image |
| `data/aqua20_clip_vit_b32_spatial_grid_aug/train_features.pt` | B/32 spatial [768,7,7], 4 views/image |
| `data/dual_clip_pooled_features.pt` | CLS tokens for B/16 and B/32 (clean only) |

The `aug` suffix means 3 random crop augmentations were extracted per image at build time
(in addition to the clean view), giving 4 views per training image total.
The test split is always clean (1 view per image).

## Custom Dataset Path

```bash
# Use images already on disk
export AQUA20_ROOT=/path/to/aqua20
python3 scripts/build_clip_features.py
```
