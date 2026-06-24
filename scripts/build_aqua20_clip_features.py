#!/usr/bin/env python3
"""Build an AQUA20 dataset of CLIP image features plus labels."""

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode AQUA20 images with CLIP and save feature tensors with labels."
    )
    parser.add_argument(
        "--aqua20-root",
        default=None,
        help="AQUA20 ImageFolder root with train/test splits (default: project data/aqua20).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for CLIP feature dataset.",
    )
    parser.add_argument("--model", default="ViT-B-16", help="OpenCLIP model name.")
    parser.add_argument("--pretrained", default="openai", help="OpenCLIP pretrained weights.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--augment-train",
        action="store_true",
        help="Apply AQUA20-style image augmentations before CLIP for the train split.",
    )
    parser.add_argument(
        "--augments-per-image",
        type=int,
        default=1,
        help="Number of augmented train views to encode for each source image.",
    )
    parser.add_argument(
        "--include-clean-train",
        action="store_true",
        help="Also include one clean CLIP-preprocessed train view before augmented views.",
    )
    parser.add_argument(
        "--random-erasing",
        type=float,
        default=0.25,
        help="Random erasing probability for augmented train views.",
    )
    parser.add_argument(
        "--feature-kind",
        choices=("pooled", "spatial_tokens", "spatial_grid"),
        default="pooled",
        help=(
            "pooled saves CLIP's final image vector; spatial_tokens saves "
            "[num_patches, width]; spatial_grid saves [width, h, w]."
        ),
    )
    parser.add_argument(
        "--project-spatial",
        action="store_true",
        help="Project spatial ViT tokens from transformer width to CLIP embed dim.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Save raw CLIP features instead of L2-normalized features.",
    )
    args = parser.parse_args()
    if args.out_dir is None:
        folder_name = "aqua20_clip_features"
        if args.feature_kind != "pooled":
            model_name = args.model.lower().replace("/", "_").replace("-", "_")
            model_name = model_name.replace("vit_b_16", "vit_b16").replace("vit_b_32", "vit_b32")
            folder_name = f"aqua20_clip_{model_name}_{args.feature_kind}"
        if args.augment_train:
            folder_name = f"{folder_name}_aug"
        args.out_dir = str(PROJECT_ROOT / "data" / folder_name)
    return args


def load_open_clip(model_name, pretrained, device):
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "This script requires open_clip_torch. Install it with: "
            "pip install open_clip_torch"
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    model.requires_grad_(False)
    return model, preprocess


def build_clip_train_augment_transform():
    import torchvision.transforms as transforms
    from timm.data.auto_augment import rand_augment_transform

    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        rand_augment_transform("rand-m9-mstd0.5-inc1", hparams={}),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std),
    ])


def encode_clip_spatial_tokens(model, images, project_spatial=False):
    import torch

    visual = model.visual
    if not all(hasattr(visual, name) for name in ("conv1", "class_embedding", "positional_embedding")):
        raise TypeError("Spatial extraction currently supports OpenCLIP ViT visual backbones.")

    x = visual.conv1(images)
    grid_h, grid_w = x.shape[-2:]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

    class_embedding = visual.class_embedding.to(dtype=x.dtype, device=x.device)
    class_embedding = class_embedding + x.new_zeros(x.shape[0], 1, x.shape[-1])
    x = torch.cat([class_embedding, x], dim=1)
    x = x + visual.positional_embedding.to(dtype=x.dtype, device=x.device)

    if hasattr(visual, "patch_dropout"):
        x = visual.patch_dropout(x)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)

    x = visual.ln_post(x)
    tokens = x[:, 1:, :]
    if project_spatial and getattr(visual, "proj", None) is not None:
        tokens = tokens @ visual.proj
    return tokens, (grid_h, grid_w)


def encode_images(model, images, args):
    import torch.nn.functional as F

    if args.feature_kind == "pooled":
        features = model.encode_image(images)
        spatial_shape = None
    else:
        features, spatial_shape = encode_clip_spatial_tokens(
            model,
            images,
            project_spatial=args.project_spatial,
        )
        if args.feature_kind == "spatial_grid":
            grid_h, grid_w = spatial_shape
            features = features.permute(0, 2, 1).reshape(
                features.shape[0],
                features.shape[-1],
                grid_h,
                grid_w,
            )

    if not args.no_normalize:
        features = F.normalize(features, dim=1 if args.feature_kind == "spatial_grid" else -1)
    return features, spatial_shape


def build_split(split_name, split_dir, out_dir, model, preprocess, device, args):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as transforms
    from tqdm import tqdm

    class ImageFolderWithPaths(ImageFolder):
        def __init__(self, root, transform=None, clean_transform=None, repeats=1, include_clean=False):
            super().__init__(root, transform=transform)
            self.clean_transform = clean_transform
            self.repeats = repeats
            self.include_clean = include_clean

        def __len__(self):
            return len(self.samples) * self.repeats

        def __getitem__(self, index):
            source_index = index % len(self.samples)
            view_index = index // len(self.samples)
            path, label = self.samples[source_index]
            image = self.loader(path)
            if self.include_clean and view_index == 0:
                image = self.clean_transform(image)
            elif self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                label = self.target_transform(label)
            return image, label, path, view_index

    if split_name == "train" and args.augment_train:
        transform = build_clip_train_augment_transform()
        if args.random_erasing > 0:
            transform = transforms.Compose([
                transform,
                transforms.RandomErasing(p=args.random_erasing),
            ])
        repeats = args.augments_per_image + int(args.include_clean_train)
        dataset = ImageFolderWithPaths(
            split_dir,
            transform=transform,
            clean_transform=preprocess,
            repeats=repeats,
            include_clean=args.include_clean_train,
        )
    else:
        dataset = ImageFolderWithPaths(split_dir, transform=preprocess)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    features = []
    labels = []
    paths = []
    view_indices = []
    spatial_shape = None
    with torch.inference_mode():
        for images, targets, batch_paths, batch_view_indices in tqdm(loader, desc=f"CLIP {split_name}"):
            images = images.to(device, non_blocking=device.type == "cuda")
            image_features, batch_spatial_shape = encode_images(model, images, args)
            spatial_shape = batch_spatial_shape or spatial_shape
            features.append(image_features.cpu())
            labels.append(targets.cpu())
            paths.extend(batch_paths)
            view_indices.extend([int(view_index) for view_index in batch_view_indices])

    feature_tensor = torch.cat(features, dim=0)
    label_tensor = torch.cat(labels, dim=0).long()
    payload = {
        "features": feature_tensor,
        "labels": label_tensor,
        "paths": paths,
        "view_indices": view_indices,
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
        "clip_model": args.model,
        "clip_pretrained": args.pretrained,
        "feature_kind": args.feature_kind,
        "spatial_shape": spatial_shape,
        "project_spatial": args.project_spatial,
        "augment_train": split_name == "train" and args.augment_train,
        "augments_per_image": args.augments_per_image if split_name == "train" and args.augment_train else 0,
        "include_clean_train": args.include_clean_train if split_name == "train" and args.augment_train else False,
        "normalized": not args.no_normalize,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / f"{split_name}_features.pt")
    with open(out_dir / f"{split_name}_metadata.jsonl", "w", encoding="utf-8") as handle:
        for index, (path, label, view_index) in enumerate(zip(paths, label_tensor.tolist(), view_indices)):
            handle.write(json.dumps({
                "index": index,
                "path": path,
                "view_index": view_index,
                "label": label,
                "class_name": dataset.classes[label],
            }) + "\n")

    print(
        f"Saved {split_name}: {tuple(feature_tensor.shape)} features -> "
        f"{out_dir / f'{split_name}_features.pt'}"
    )


def main():
    args = parse_args()
    import torch

    from spiral_project.config import get_dataset_config
    from spiral_project.data import prepare_aqua20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_dataset_config("aqua20")
    aqua20_root = os.path.abspath(args.aqua20_root or config["data_dir"])
    train_dir, test_dir = prepare_aqua20(aqua20_root)

    model, preprocess = load_open_clip(args.model, args.pretrained, device)
    out_dir = Path(args.out_dir).resolve()
    build_split("train", train_dir, out_dir, model, preprocess, device, args)
    build_split("test", test_dir, out_dir, model, preprocess, device, args)

    summary = {
        "aqua20_root": aqua20_root,
        "out_dir": str(out_dir),
        "clip_model": args.model,
        "clip_pretrained": args.pretrained,
        "feature_kind": args.feature_kind,
        "project_spatial": args.project_spatial,
        "augment_train": args.augment_train,
        "augments_per_image": args.augments_per_image if args.augment_train else 0,
        "include_clean_train": args.include_clean_train if args.augment_train else False,
        "random_erasing": args.random_erasing if args.augment_train else 0.0,
        "normalized": not args.no_normalize,
    }
    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Done. CLIP feature dataset saved to {out_dir}")


if __name__ == "__main__":
    main()
