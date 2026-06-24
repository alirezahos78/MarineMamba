# Spiral Vision Mamba

Training code for a spiral-scan Mamba vision model configured for the AQUA20 underwater-species dataset.

## Project Layout

```text
spiral_final/
├── main.py
├── README.md
├── requirements.txt
├── BASELINES.md
├── baselines/
│   ├── efficient_vmamba/
│   │   ├── logs/
│   │   └── scripts/
│   └── vim/
│       ├── logs/
│       └── scripts/
├── spiral_project/
│   ├── augmentations.py
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── model.py
│   ├── paths.py
│   ├── trainer.py
│   └── utils.py
├── data/
├── logs/
└── *.pth
```

## What This Repo Does

- trains the spiral-scan model on the fixed AQUA20 train/test split
- resizes/crops images to 224x224 and embeds non-overlapping 16x16 patches
- processes local features at 2x, 4x, and 8x downsample scales
- retains the older CIFAR and Tiny ImageNet loaders for optional experiments

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/download_aqua20.py
python3 scripts/build_aqua20_clip_features.py
python3 main.py
```

The default experiment trains a classifier on precomputed AQUA20 CLIP features.
To train the SpyMamba image model instead, run `python3 main.py --datasets aqua20`.
To train SpyMamba on spatial CLIP patch features, first generate the spatial
dataset, then run:

```bash
python3 scripts/build_aqua20_clip_features.py --model ViT-B-16 --feature-kind spatial_grid --augment-train --augments-per-image 3 --include-clean-train
python3 main.py --datasets aqua20_clip_spatial
```

This feeds each `[768, 14, 14]` CLIP ViT-B/16 feature grid through a `768 -> 128` projection
and then through SpyMamba blocks with the local 8x branch disabled. The head
classifies from average-pooled SpyMamba tokens with regular cross-entropy. The
training split stores multiple augmented CLIP feature views per image, while
the test split remains clean.

The SpyMamba image experiment trains AQUA20 at 224x224 resolution with 8x8 patches.
Each Mamba block uses three local downsample scales: 2x, 4x, and 8x.
Training uses moderate class-balanced sampling, a 3e-4 peak learning rate,
MixUp/CutMix on half of batches, and early stopping to prevent majority-class collapse.
For transfer learning, pretrain SpyMamba on a source dataset, then fine-tune AQUA20
from the saved SpyMamba checkpoint with `--pretrained-path`.

## FathomNet Pretraining

`scripts/fetch_fathomnet_pretrain.py` writes the cropped ImageFolder dataset to
the exact `--out-root` path passed on the command line. For example, this creates
`/local-scratch/.../fathomnet_cls/train/<class>/*.jpg` and
`/local-scratch/.../fathomnet_cls/val/<class>/*.jpg`:

```bash
python3 scripts/fetch_fathomnet_pretrain.py --out-root /local-scratch/.../fathomnet_cls --top-k 60 --max-per-class 800
```

Pretrain SpyMamba on that ImageFolder:

```bash
python3 main.py --datasets fathomnet_pretrain --branch-settings full --data-dir /local-scratch/.../fathomnet_cls --num-classes 60
```

Then fine-tune AQUA20 from the FathomNet checkpoint:

```bash
python3 main.py --datasets aqua20 --branch-settings full --pretrained-path best_model_fathomnet_pretrain_spiral_full_seed_42.pth
```

When `--pretrained-path` is used, AQUA20 fine-tuning freezes the first 5
SpyMamba blocks by default and leaves the last block plus the remaining model
parameters trainable. Override that with `--freeze-first-blocks-on-transfer N`.

## Grad-CAM Maps

To save Grad-CAM heatmaps and overlays from a trained checkpoint:

```bash
python3 scripts/extract_gradcam_maps.py --dataset aqua20 --branch-setting full --seed 42 --split test --index 0 --count 8
```

Outputs are written to `gradcam_maps/<run_name>/` by default. Each sample gets the original image, Grad-CAM heatmap, overlay, and a compressed `.npz` file with the raw Grad-CAM array plus prediction metadata.

## Notes

- AQUA20 downloads automatically during training, or explicitly through `scripts/download_aqua20.py`
- Tiny ImageNet is downloaded and reorganized automatically
- logs are written to `logs/`
- results are written to `spiral_results_all_datasets.json`
- checkpoints, datasets, generated JSON, and runtime logs are ignored by git
- external comparison baselines are documented in `BASELINES.md`
