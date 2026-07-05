import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


class DualCLIPFeatureDataset(torch.utils.data.Dataset):
    """
    Loads paired ViT-B/16 and ViT-B/32 spatial CLIP features.

    Always returns a 5-tuple (when cls_path is provided):
        (fine_feat, coarse_feat, fine_cls, coarse_cls, label)

    Or a 3-tuple when cls_path is None:
        (fine_feat, coarse_feat, label)
    """
    def __init__(self, fine_path, coarse_path, cls_path=None, split="train"):
        fine   = torch.load(fine_path,   map_location="cpu")
        coarse = torch.load(coarse_path, map_location="cpu")

        self.fine_features   = fine["features"].float()
        self.coarse_features = coarse["features"].float()
        fine_labels   = fine["labels"].long()
        coarse_labels = coarse["labels"].long()

        if not torch.equal(fine_labels, coarse_labels):
            raise ValueError(
                "B/16 and B/32 feature files have mismatched labels. "
                "Rebuild both with identical settings and augmentation order."
            )
        self.labels      = fine_labels
        self.classes     = fine.get("classes", [])
        self.class_to_idx = fine.get("class_to_idx", {})

        self.fine_cls   = None
        self.coarse_cls = None
        n_spatial = self.fine_features.shape[0]
        if cls_path is not None:
            cls_data       = torch.load(cls_path, map_location="cpu")[split]
            fine_cls_all   = cls_data["ViT-B-16"].float()
            coarse_cls_all = cls_data["ViT-B-32"].float()
            n_images = fine_cls_all.shape[0]
            idx = torch.arange(n_spatial) % n_images
            self.fine_cls   = fine_cls_all[idx]
            self.coarse_cls = coarse_cls_all[idx]

        self.targets  = self.labels.tolist()
        self._has_cls = self.fine_cls is not None

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, idx):
        if self._has_cls:
            return (self.fine_features[idx], self.coarse_features[idx],
                    self.fine_cls[idx], self.coarse_cls[idx],
                    self.labels[idx])
        return (self.fine_features[idx], self.coarse_features[idx],
                self.labels[idx])


def build_dataloaders(config):
    fine_train   = config["fine_train_path"]
    fine_test    = config["fine_test_path"]
    coarse_train = config["coarse_train_path"]
    coarse_test  = config["coarse_test_path"]
    cls_path     = config.get("cls_path")

    trainset = DualCLIPFeatureDataset(fine_train, coarse_train, cls_path=cls_path, split="train")
    testset  = DualCLIPFeatureDataset(fine_test,  coarse_test,  cls_path=cls_path, split="test")

    use_cuda     = torch.cuda.is_available()
    num_workers  = config.get("num_workers", 0)
    loader_kwargs = {"num_workers": num_workers, "pin_memory": use_cuda}

    mode = config.get("balanced_sampling")
    if mode in ("sqrt", "uniform"):
        targets      = torch.as_tensor(trainset.targets, dtype=torch.long)
        class_counts = torch.bincount(targets, minlength=config["num_classes"]).float()
        weights      = class_counts.rsqrt() if mode == "sqrt" else class_counts.reciprocal()
        sampler      = WeightedRandomSampler(weights[targets], num_samples=len(trainset), replacement=True)
        trainloader  = DataLoader(trainset, batch_size=config["batch_size"], sampler=sampler, **loader_kwargs)
    else:
        trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, **loader_kwargs)

    testloader = DataLoader(testset, batch_size=config["batch_size"], shuffle=False, **loader_kwargs)
    return trainloader, testloader
