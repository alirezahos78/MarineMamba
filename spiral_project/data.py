import os
import shutil
import urllib.request
import zipfile

import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from .paths import DATA_ROOT
from .utils import ensure_dir

try:
    from timm.data import RepeatedAugmentSampler
except ImportError:
    class RepeatedAugmentSampler(Sampler):
        def __init__(self, dataset, num_repeats=3, shuffle=True):
            self.dataset = dataset
            self.num_repeats = num_repeats
            self.shuffle = shuffle

        def __iter__(self):
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                indices = torch.randperm(len(self.dataset)).tolist()
            repeated = []
            for idx in indices:
                repeated.extend([idx] * self.num_repeats)
            return iter(repeated)

        def __len__(self):
            return len(self.dataset) * self.num_repeats


def download_file(url, destination):
    if os.path.exists(destination):
        return
    ensure_dir(os.path.dirname(destination))
    print(f"⬇️ Downloading: {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"✅ Downloaded to: {destination}")


def prepare_tiny_imagenet(data_dir):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        return train_dir, val_dir

    ensure_dir(DATA_ROOT)
    archive_path = os.path.join(DATA_ROOT, "tiny-imagenet-200.zip")
    download_file("http://cs231n.stanford.edu/tiny-imagenet-200.zip", archive_path)

    if not os.path.isdir(data_dir):
        print("📦 Extracting Tiny ImageNet...")
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(DATA_ROOT)

    val_images_dir = os.path.join(val_dir, "images")
    annotations_path = os.path.join(val_dir, "val_annotations.txt")
    if os.path.isdir(val_images_dir) and os.path.isfile(annotations_path):
        print("🗂️ Preparing Tiny ImageNet validation folders...")
        with open(annotations_path, "r") as handle:
            for line in handle:
                image_name, class_id = line.strip().split("\t")[:2]
                class_dir = os.path.join(val_dir, class_id)
                class_images_dir = os.path.join(class_dir, "images")
                ensure_dir(class_images_dir)

                src = os.path.join(val_images_dir, image_name)
                dst = os.path.join(class_images_dir, image_name)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.move(src, dst)

        if os.path.isdir(val_images_dir) and not os.listdir(val_images_dir):
            os.rmdir(val_images_dir)

    return train_dir, val_dir


def prepare_aqua20(data_dir):
    """Download AQUA20 and materialize its fixed splits for ImageFolder."""
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    complete_marker = os.path.join(data_dir, ".complete")
    if os.path.isfile(complete_marker):
        return train_dir, test_dir

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "AQUA20 download requires the 'datasets' package. "
            "Install the project requirements first."
        ) from exc

    print("⬇️ Downloading AQUA20 from Hugging Face...")
    dataset = load_dataset("taufiktrf/AQUA20")
    class_names = dataset["train"].features["label"].names

    for split_name, split_dir in (("train", train_dir), ("test", test_dir)):
        for class_name in class_names:
            ensure_dir(os.path.join(split_dir, class_name))

        print(f"📦 Preparing AQUA20 {split_name} split...")
        for index, example in enumerate(dataset[split_name]):
            label = int(example["label"])
            image_path = os.path.join(split_dir, class_names[label], f"{index:06d}.jpg")
            if not os.path.isfile(image_path):
                example["image"].convert("RGB").save(image_path, quality=95)

    ensure_dir(data_dir)
    with open(complete_marker, "w", encoding="utf-8") as marker:
        marker.write("AQUA20: 6559 train, 1612 test\n")

    return train_dir, test_dir


def prepare_imagefolder_splits(data_dir):
    train_dir = os.path.join(data_dir, "train")
    eval_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(data_dir, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Expected ImageFolder training split at {train_dir}")
    if not os.path.isdir(eval_dir):
        raise FileNotFoundError(f"Expected ImageFolder val or test split under {data_dir}")
    return train_dir, eval_dir


class ClipFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features_path):
        payload = torch.load(features_path, map_location="cpu")
        self.features = payload["features"].float()
        self.labels = payload["labels"].long()
        self.targets = self.labels.tolist()
        self.paths = payload.get("paths", [])
        self.classes = payload.get("classes", [])
        self.class_to_idx = payload.get("class_to_idx", {})

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def prepare_clip_feature_splits(data_dir):
    train_path = os.path.join(data_dir, "train_features.pt")
    test_path = os.path.join(data_dir, "test_features.pt")
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Expected CLIP train features at {train_path}")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Expected CLIP test features at {test_path}")
    return train_path, test_path


def build_transforms(dataset_name, config):
    import torchvision.transforms as transforms
    from timm.data.auto_augment import rand_augment_transform

    if dataset_name in {"aqua20", "fathomnet_pretrain"}:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config["img_size"], scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform("rand-m9-mstd0.5-inc1", hparams={})
            if config.get("randaugment", False) else transforms.Lambda(lambda image: image),
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"]),
            transforms.RandomErasing(p=config.get("random_erasing", 0.0)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.get("resize_size", config["img_size"])),
            transforms.CenterCrop(config["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"]),
        ])
        return train_transform, test_transform

    transform_list = [
        transforms.RandomCrop(config["img_size"], padding=config["img_size"] // 8),
        transforms.RandomHorizontalFlip(),
    ]
    if config.get("randaugment", False):
        transform_list.append(rand_augment_transform("rand-m9-mstd0.5-inc1", hparams={}))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
        transforms.RandomErasing(p=config.get("random_erasing", 0.0)),
    ])
    train_transform = transforms.Compose(transform_list)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
    ])
    return train_transform, test_transform


def load_datasets(dataset_name, config):
    if config.get("feature_dataset"):
        train_path, test_path = prepare_clip_feature_splits(config["data_dir"])
        return ClipFeatureDataset(train_path), ClipFeatureDataset(test_path)

    from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

    train_transform, test_transform = build_transforms(dataset_name, config)

    if dataset_name == "aqua20":
        train_dir, test_dir = prepare_aqua20(config["data_dir"])
        trainset = ImageFolder(root=train_dir, transform=train_transform)
        testset = ImageFolder(root=test_dir, transform=test_transform)
    elif dataset_name == "fathomnet_pretrain":
        train_dir, val_dir = prepare_imagefolder_splits(config["data_dir"])
        trainset = ImageFolder(root=train_dir, transform=train_transform)
        testset = ImageFolder(root=val_dir, transform=test_transform)
    elif dataset_name == "cifar10":
        trainset = CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_transform)
        testset = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_transform)
    elif dataset_name == "cifar100":
        trainset = CIFAR100(root=DATA_ROOT, train=True, download=True, transform=train_transform)
        testset = CIFAR100(root=DATA_ROOT, train=False, download=True, transform=test_transform)
    elif dataset_name == "tiny_imagenet":
        train_dir, val_dir = prepare_tiny_imagenet(config["data_dir"])
        trainset = ImageFolder(root=train_dir, transform=train_transform)
        testset = ImageFolder(root=val_dir, transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return trainset, testset


def build_dataloaders(dataset_name, config):
    trainset, testset = load_datasets(dataset_name, config)
    use_cuda = torch.cuda.is_available()
    num_workers = config.get("num_workers", 4)
    dataloader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = config.get("persistent_workers", False)
        dataloader_kwargs["prefetch_factor"] = config.get("prefetch_factor", 2)

    if config.get("balanced_sampling"):
        targets = torch.as_tensor(trainset.targets, dtype=torch.long)
        class_counts = torch.bincount(targets, minlength=config["num_classes"]).float()
        if torch.any(class_counts == 0):
            raise ValueError("Balanced sampling requires at least one training image per class")

        sampling_mode = config["balanced_sampling"]
        if sampling_mode == "sqrt":
            class_weights = class_counts.rsqrt()
        elif sampling_mode == "uniform":
            class_weights = class_counts.reciprocal()
        else:
            raise ValueError(f"Unknown balanced_sampling mode: {sampling_mode}")

        sampler = WeightedRandomSampler(
            weights=class_weights[targets],
            num_samples=len(trainset),
            replacement=True,
        )
        trainloader = DataLoader(
            trainset,
            batch_size=config["batch_size"],
            sampler=sampler,
            **dataloader_kwargs,
        )
    elif config.get("repeated_aug", False) and dataset_name != "tiny_imagenet":
        sampler = RepeatedAugmentSampler(trainset, num_repeats=3)
        trainloader = DataLoader(
            trainset,
            batch_size=config["batch_size"],
            sampler=sampler,
            **dataloader_kwargs,
        )
    else:
        trainloader = DataLoader(
            trainset,
            batch_size=config["batch_size"],
            shuffle=True,
            **dataloader_kwargs,
        )

    testloader = DataLoader(
        testset,
        batch_size=config["batch_size"],
        shuffle=False,
        **dataloader_kwargs,
    )
    return trainloader, testloader
