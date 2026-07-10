#!/usr/bin/env python3
"""
Build all CLIP feature caches needed by MarineMamba — fully standalone, no MarineMamba dependency.

Produces three caches inside marinemamba/data/:
  aqua20_clip_vit_b16_spatial_grid_aug/   ← ViT-B/16 spatial patches (14×14)
  aqua20_clip_vit_b32_spatial_grid_aug/   ← ViT-B/32 spatial patches (7×7)
  dual_clip_pooled_features.pt            ← B/16 + B/32 CLS tokens (clean, 1 view)

Usage:
    python3 scripts/build_clip_features.py --aqua20-root /path/to/aqua20

The aqua20 directory must contain train/ and test/ sub-folders
(one sub-folder per class, ImageFolder layout).
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from marinemamba.paths import AQUA20_ROOT, B16_FEATURES_ROOT, B32_FEATURES_ROOT, CLS_FEATURES_PATH


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aqua20-root", default=AQUA20_ROOT,
                   help="Path to dataset root (must have train/ and test/ sub-dirs).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    # Augmentation for spatial features
    p.add_argument("--augments-per-image", type=int, default=3,
                   help="Number of augmented views per image for train split")
    p.add_argument("--include-clean-train", action="store_true", default=True,
                   help="Always include one clean view in train (view index 0)")
    p.add_argument("--random-erasing", type=float, default=0.25)
    # Which caches to build
    p.add_argument("--skip-b16", action="store_true", help="Skip building B/16 spatial cache")
    p.add_argument("--skip-b32", action="store_true", help="Skip building B/32 spatial cache")
    p.add_argument("--skip-cls", action="store_true", help="Skip building CLS pooled cache")
    # Custom output directories (override defaults)
    p.add_argument("--out-b16", default=None, help="Output dir for B/16 spatial cache")
    p.add_argument("--out-b32", default=None, help="Output dir for B/32 spatial cache")
    p.add_argument("--out-cls", default=None, help="Output path for CLS pooled features (.pt)")
    return p.parse_args()


def load_clip(model_name, device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai", device=device,
    )
    model.eval()
    model.requires_grad_(False)
    return model, preprocess


def build_augment_transform():
    import torchvision.transforms as T
    from timm.data.auto_augment import rand_augment_transform
    mean = (0.48145466, 0.4578275, 0.40821073)
    std  = (0.26862954, 0.26130258, 0.27577711)
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        rand_augment_transform("rand-m9-mstd0.5-inc1", hparams={}),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def encode_spatial_grid(visual, images):
    """Extract [B, C, H, W] spatial patch features from a ViT CLIP visual encoder."""
    import torch
    import torch.nn.functional as F

    x = visual.conv1(images)
    grid_h, grid_w = x.shape[-2:]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x) + x.new_zeros(x.shape[0], 1, x.shape[-1])
    x = torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x[:, 1:, :])  # patch tokens only: [B, H*W, C]
    feats = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], grid_h, grid_w)
    feats = F.normalize(feats, dim=1)
    return feats.float(), (grid_h, grid_w)


def build_spatial_split(split, split_dir, out_dir, model, preprocess, device, args):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as T
    from tqdm import tqdm

    class AugFolder(ImageFolder):
        def __init__(self, root, transform, clean_transform=None, repeats=1, include_clean=False):
            super().__init__(root, transform=transform)
            self.clean_transform = clean_transform
            self.repeats = repeats
            self.include_clean = include_clean

        def __len__(self):
            return len(self.samples) * self.repeats

        def __getitem__(self, index):
            src_idx  = index % len(self.samples)
            view_idx = index // len(self.samples)
            path, label = self.samples[src_idx]
            img = self.loader(path)
            if self.include_clean and view_idx == 0:
                img = self.clean_transform(img)
            elif self.transform is not None:
                img = self.transform(img)
            return img, label, path, view_idx

    if split == "train":
        aug = build_augment_transform()
        if args.random_erasing > 0:
            aug = T.Compose([aug, T.RandomErasing(p=args.random_erasing)])
        repeats = args.augments_per_image + int(args.include_clean_train)
        dataset = AugFolder(split_dir, transform=aug, clean_transform=preprocess,
                            repeats=repeats, include_clean=args.include_clean_train)
    else:
        dataset = AugFolder(split_dir, transform=preprocess)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")

    feats_list, labels_list, paths_list, views_list = [], [], [], []
    spatial_shape = None

    with torch.inference_mode():
        for images, targets, batch_paths, batch_views in tqdm(loader, desc=f"  {split}"):
            images = images.to(device, non_blocking=device.type == "cuda")
            feats, s_shape = encode_spatial_grid(model.visual, images)
            spatial_shape = s_shape
            feats_list.append(feats.cpu())
            labels_list.append(targets)
            paths_list.extend(batch_paths)
            views_list.extend(batch_views.tolist())

    feature_tensor = torch.cat(feats_list, dim=0)
    label_tensor   = torch.cat(labels_list, dim=0).long()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": feature_tensor,
        "labels": label_tensor,
        "paths": paths_list,
        "view_indices": views_list,
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
        "spatial_shape": spatial_shape,
    }
    torch.save(payload, out_dir / f"{split}_features.pt")

    with open(out_dir / f"{split}_metadata.jsonl", "w") as f:
        for i, (p, l, v) in enumerate(zip(paths_list, label_tensor.tolist(), views_list)):
            f.write(json.dumps({"index": i, "path": p, "label": l,
                                "class_name": dataset.classes[l], "view_index": v}) + "\n")

    print(f"  Saved {split}: {tuple(feature_tensor.shape)} → {out_dir / f'{split}_features.pt'}")
    return dataset.classes


def build_cls_features(train_dir, test_dir, b16_model, b32_model, preprocess, device, args, out_path=None):
    """Build clean CLS pooled features for both B/16 and B/32 (1 view per image)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from tqdm import tqdm

    out = {}
    for split, split_dir in [("train", train_dir), ("test", test_dir)]:
        dataset = ImageFolder(split_dir, transform=preprocess)
        loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=device.type == "cuda")

        b16_list, b32_list, labels_list = [], [], []
        with torch.inference_mode():
            for images, targets in tqdm(loader, desc=f"  CLS {split}"):
                images = images.to(device, non_blocking=device.type == "cuda")
                cls16 = F.normalize(b16_model.encode_image(images).float(), dim=-1)
                cls32 = F.normalize(b32_model.encode_image(images).float(), dim=-1)
                b16_list.append(cls16.cpu())
                b32_list.append(cls32.cpu())
                labels_list.append(targets)

        out[split] = {
            "ViT-B-16": torch.cat(b16_list, dim=0),
            "ViT-B-32": torch.cat(b32_list, dim=0),
            "labels":   torch.cat(labels_list, dim=0).long(),
            "classes":  dataset.classes,
        }
        n = out[split]["labels"].shape[0]
        print(f"  Saved CLS {split}: {n} samples (B/16: {tuple(out[split]['ViT-B-16'].shape)}, "
              f"B/32: {tuple(out[split]['ViT-B-32'].shape)})")

    save_path = out_path or CLS_FEATURES_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(out, save_path)
    print(f"  → {save_path}")


def main():
    args = parse_args()
    import torch

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = os.path.join(args.aqua20_root, "train")
    test_dir  = os.path.join(args.aqua20_root, "test")

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"train/test directories not found under {args.aqua20_root}.\n"
            "Expected layout: <root>/train/<class>/*.jpg  and  <root>/test/<class>/*.jpg"
        )

    out_b16 = args.out_b16 or B16_FEATURES_ROOT
    out_b32 = args.out_b32 or B32_FEATURES_ROOT
    out_cls = args.out_cls or CLS_FEATURES_PATH

    print(f"Device: {device}")
    print(f"Dataset root : {args.aqua20_root}")
    print(f"Out B/16     : {out_b16}")
    print(f"Out B/32     : {out_b32}")
    print(f"Out CLS      : {out_cls}")

    if not args.skip_b16:
        print("\n[1/3] Building ViT-B/16 spatial features (14×14) ...")
        b16_model, preprocess = load_clip("ViT-B-16", device)
        build_spatial_split("train", train_dir, out_b16, b16_model, preprocess, device, args)
        build_spatial_split("test",  test_dir,  out_b16, b16_model, preprocess, device, args)
        with open(os.path.join(out_b16, "dataset_info.json"), "w") as f:
            json.dump({"clip_model": "ViT-B-16", "spatial_shape": [14, 14],
                       "augments_per_image": args.augments_per_image}, f, indent=2)
    else:
        print("\n[1/3] Skipping B/16 spatial features.")
        b16_model, preprocess = load_clip("ViT-B-16", device)

    if not args.skip_b32:
        print("\n[2/3] Building ViT-B/32 spatial features (7×7) ...")
        b32_model, _ = load_clip("ViT-B-32", device)
        build_spatial_split("train", train_dir, out_b32, b32_model, preprocess, device, args)
        build_spatial_split("test",  test_dir,  out_b32, b32_model, preprocess, device, args)
        with open(os.path.join(out_b32, "dataset_info.json"), "w") as f:
            json.dump({"clip_model": "ViT-B-32", "spatial_shape": [7, 7],
                       "augments_per_image": args.augments_per_image}, f, indent=2)
    else:
        print("\n[2/3] Skipping B/32 spatial features.")
        b32_model, _ = load_clip("ViT-B-32", device)

    if not args.skip_cls:
        print("\n[3/3] Building CLS pooled features (clean, 1 view per image) ...")
        build_cls_features(train_dir, test_dir, b16_model, b32_model, preprocess, device, args,
                           out_path=out_cls)
    else:
        print("\n[3/3] Skipping CLS pooled features.")

    print("\nAll feature caches built.")
    print(f"  Run: python3 main.py")


if __name__ == "__main__":
    main()
