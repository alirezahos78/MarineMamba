import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


def augment_minority_classes_dual(
    fine_features,
    coarse_features,
    labels,
    num_classes,
    min_threshold=None,
    target_samples=None,
    alpha_range=(0.3, 0.7),
    noise_scale=0.0,
):
    """
    Intra-class MixUp in CLIP feature space, applied jointly to both
    fine (B/16) and coarse (B/32) features so each (fine, coarse) pair
    remains consistent after augmentation.
    """
    class_indices = [torch.where(labels == c)[0] for c in range(num_classes)]
    class_counts = torch.tensor([len(idx) for idx in class_indices], dtype=torch.float)

    if min_threshold is None:
        min_threshold = int(class_counts.mean().item())
    if target_samples is None:
        target_samples = int(class_counts.mean().item())

    syn_fine, syn_coarse, syn_labels = [], [], []

    for c in range(num_classes):
        indices = class_indices[c]
        n = len(indices)
        if n == 0 or n >= min_threshold:
            continue

        n_needed = max(0, target_samples - n)
        cf = fine_features[indices]
        cc = coarse_features[indices]

        for _ in range(n_needed):
            i, j = torch.randint(n, (2,))
            alpha = float(torch.empty(1).uniform_(*alpha_range))

            sf = alpha * cf[i] + (1.0 - alpha) * cf[j]
            sc = alpha * cc[i] + (1.0 - alpha) * cc[j]

            if noise_scale > 0.0:
                sf = sf + torch.randn_like(sf) * noise_scale
                sc = sc + torch.randn_like(sc) * noise_scale

            sf = F.normalize(sf.unsqueeze(0), dim=1).squeeze(0)
            sc = F.normalize(sc.unsqueeze(0), dim=1).squeeze(0)

            syn_fine.append(sf)
            syn_coarse.append(sc)
            syn_labels.append(c)

    if not syn_fine:
        return fine_features, coarse_features, labels

    aug_fine = torch.cat([fine_features, torch.stack(syn_fine)], dim=0)
    aug_coarse = torch.cat([coarse_features, torch.stack(syn_coarse)], dim=0)
    aug_labels = torch.cat([labels, torch.tensor(syn_labels, dtype=labels.dtype)], dim=0)

    minority_classes = int((class_counts < min_threshold).sum().item())
    print(
        f"[FeatureAug] Added {len(syn_labels)} synthetic pairs across "
        f"{minority_classes} minority classes "
        f"(threshold={min_threshold}, target={target_samples}). "
        f"Dataset: {len(labels)} → {len(aug_labels)} samples."
    )
    return aug_fine, aug_coarse, aug_labels


class DualCLIPFeatureDataset(torch.utils.data.Dataset):
    """
    Loads paired ViT-B/16 and ViT-B/32 spatial CLIP features.
    Returns (fine_feat, coarse_feat, label) per sample.

    Optionally loads CLS pooled features (512-d each) from a separate file.
    When cls_path is set, returns (fine_feat, coarse_feat, fine_cls, coarse_cls, label).
    CLS features are aligned to spatial features via modulo indexing
    (one CLS feature per unique image, repeated across augmented views).
    """
    def __init__(self, fine_path, coarse_path, feature_aug_config=None, cls_path=None, split="train"):
        fine = torch.load(fine_path, map_location="cpu")
        coarse = torch.load(coarse_path, map_location="cpu")

        self.fine_features = fine["features"].float()
        self.coarse_features = coarse["features"].float()
        fine_labels = fine["labels"].long()
        coarse_labels = coarse["labels"].long()

        if not torch.equal(fine_labels, coarse_labels):
            raise ValueError(
                "B/16 and B/32 feature files have mismatched labels. "
                "Rebuild both with identical settings and augmentation order."
            )
        self.labels = fine_labels
        self.classes = fine.get("classes", [])
        self.class_to_idx = fine.get("class_to_idx", {})

        # Optional CLS bypass features
        self.fine_cls = None
        self.coarse_cls = None
        if cls_path is not None:
            cls_data = torch.load(cls_path, map_location="cpu")[split]
            fine_cls_all   = cls_data["ViT-B-16"].float()  # [N_images, 512]
            coarse_cls_all = cls_data["ViT-B-32"].float()
            n_images = fine_cls_all.shape[0]
            # Spatial may have multiple augmented views per image; map back via modulo
            n_spatial = self.fine_features.shape[0]
            idx = torch.arange(n_spatial) % n_images
            self.fine_cls   = fine_cls_all[idx]    # [n_spatial, 512]
            self.coarse_cls = coarse_cls_all[idx]

        if feature_aug_config:
            num_classes = len(self.classes) if self.classes else int(self.labels.max().item()) + 1
            self.fine_features, self.coarse_features, self.labels = augment_minority_classes_dual(
                self.fine_features,
                self.coarse_features,
                self.labels,
                num_classes=num_classes,
                **feature_aug_config,
            )
            # Expand CLS features to match augmented size
            if self.fine_cls is not None:
                n_orig = fine_cls_all.shape[0] if cls_path else self.fine_cls.shape[0]
                n_new  = self.labels.shape[0]
                # Augmented samples are appended after originals; map them via modulo too
                idx_aug = torch.arange(n_new) % n_orig
                # For the original samples use direct indexing; for synthetic use random
                orig_fine_cls = self.fine_cls[:n_orig] if n_orig <= self.fine_cls.shape[0] else self.fine_cls
                self.fine_cls   = orig_fine_cls[torch.arange(n_new) % orig_fine_cls.shape[0]]
                self.coarse_cls = coarse_cls_all[torch.arange(n_new) % coarse_cls_all.shape[0]] if cls_path else self.coarse_cls

        self.targets = self.labels.tolist()
        self._has_cls = self.fine_cls is not None

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, idx):
        if self._has_cls:
            return (self.fine_features[idx], self.coarse_features[idx],
                    self.fine_cls[idx], self.coarse_cls[idx], self.labels[idx])
        return self.fine_features[idx], self.coarse_features[idx], self.labels[idx]


def build_dataloaders(config):
    fine_train = config["fine_train_path"]
    fine_test = config["fine_test_path"]
    coarse_train = config["coarse_train_path"]
    coarse_test = config["coarse_test_path"]

    feature_aug_config = None
    if config.get("feature_aug"):
        feature_aug_config = {
            "min_threshold": config.get("feature_aug_min_threshold"),
            "target_samples": config.get("feature_aug_target_samples"),
            "alpha_range": config.get("feature_aug_alpha_range", (0.3, 0.7)),
            "noise_scale": config.get("feature_aug_noise_scale", 0.0),
        }

    cls_path = config.get("cls_path")
    trainset = DualCLIPFeatureDataset(fine_train, coarse_train, feature_aug_config,
                                      cls_path=cls_path, split="train")
    testset  = DualCLIPFeatureDataset(fine_test, coarse_test,
                                      cls_path=cls_path, split="test")

    use_cuda = torch.cuda.is_available()
    num_workers = config.get("num_workers", 0)
    loader_kwargs = {"num_workers": num_workers, "pin_memory": use_cuda}

    if config.get("balanced_sampling"):
        targets = torch.as_tensor(trainset.targets, dtype=torch.long)
        class_counts = torch.bincount(targets, minlength=config["num_classes"]).float()
        mode = config["balanced_sampling"]
        if mode == "sqrt":
            weights = class_counts.rsqrt()
        elif mode == "uniform":
            weights = class_counts.reciprocal()
        else:
            raise ValueError(f"Unknown balanced_sampling mode: {mode}")
        sampler = WeightedRandomSampler(weights[targets], num_samples=len(trainset), replacement=True)
        trainloader = DataLoader(trainset, batch_size=config["batch_size"], sampler=sampler, **loader_kwargs)
    else:
        trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, **loader_kwargs)

    testloader = DataLoader(testset, batch_size=config["batch_size"], shuffle=False, **loader_kwargs)
    return trainloader, testloader
