import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision import transforms


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from spiral_project.config import get_dataset_config  # noqa: E402
from spiral_project.data import prepare_aqua20, prepare_tiny_imagenet  # noqa: E402
from spiral_project.paths import DATA_ROOT  # noqa: E402
from spiral_project.trainer import build_model, make_run_name  # noqa: E402

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR


def parse_args():
    parser = argparse.ArgumentParser(description="Create Grad-CAM visualizations for Spiral Vision Mamba checkpoints.")
    parser.add_argument("--dataset", default="aqua20", choices=["aqua20", "cifar10", "cifar100", "tiny_imagenet"])
    parser.add_argument("--branch-setting", default="full", choices=["full", "no_local_2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None, help="Defaults to best_model_<run_name>.pth.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--index", type=int, default=0, help="First dataset index to visualize.")
    parser.add_argument("--count", type=int, default=8, help="Number of images to visualize.")
    parser.add_argument("--target-class", type=int, default=None, help="Class id for Grad-CAM. Defaults to predicted class.")
    parser.add_argument("--output-dir", default=None, help="Defaults to gradcam_maps/<run_name>.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay opacity.")
    return parser.parse_args()


def make_dataset(dataset_name, config, split):
    transform = transforms.Compose(
        [
            transforms.Resize(config["img_size"]),
            transforms.CenterCrop(config["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"]),
        ]
    )
    is_train = split == "train"
    if dataset_name == "cifar10":
        return CIFAR10(root=DATA_ROOT, train=is_train, download=False, transform=transform)
    if dataset_name == "cifar100":
        return CIFAR100(root=DATA_ROOT, train=is_train, download=False, transform=transform)

    if dataset_name == "aqua20":
        train_dir, test_dir = prepare_aqua20(config["data_dir"])
        return ImageFolder(root=train_dir if is_train else test_dir, transform=transform)

    train_dir, val_dir = prepare_tiny_imagenet(config["data_dir"])
    return ImageFolder(root=train_dir if is_train else val_dir, transform=transform)


def denormalize_image(tensor, mean, std):
    mean = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    image = (tensor.cpu() * std + mean).clamp(0, 1)
    image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(image)


def colorize_cam(cam, image_size):
    cam = cam.astype(np.float32)
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    gray = Image.fromarray((cam * 255).astype(np.uint8), mode="L")
    gray = gray.resize(image_size, resample=RESAMPLE_BILINEAR)
    cam = np.asarray(gray).astype(np.float32) / 255.0

    heatmap = np.zeros((*cam.shape, 3), dtype=np.float32)
    heatmap[..., 0] = np.clip(1.5 * cam, 0, 1)
    heatmap[..., 1] = np.clip(1.5 * (1.0 - np.abs(cam - 0.55) * 2.0), 0, 1)
    heatmap[..., 2] = np.clip(1.5 * (1.0 - cam), 0, 1)
    return Image.fromarray((heatmap * 255).astype(np.uint8))


def overlay_heatmap(image, heatmap, alpha):
    image_np = np.asarray(image).astype(np.float32)
    heatmap_np = np.asarray(heatmap).astype(np.float32)
    overlay = (1.0 - alpha) * image_np + alpha * heatmap_np
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


class PatchGradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.handle = model.patch_embed[0].register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output
        output.retain_grad()

    def remove(self):
        self.handle.remove()

    def __call__(self, image, target_class=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        pred = int(logits.argmax(dim=1).item())
        class_id = pred if target_class is None else target_class
        score = logits[:, class_id].sum()
        score.backward()

        gradients = self.activations.grad
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)[0].detach().cpu().numpy()
        prob = float(torch.softmax(logits.detach(), dim=1)[0, pred].item())
        return cam, pred, prob, class_id


def main():
    args = parse_args()
    config = get_dataset_config(args.dataset)
    run_name = make_run_name(args.dataset, args.branch_setting, args.seed)
    checkpoint = args.checkpoint or os.path.join(PROJECT_ROOT, f"best_model_{run_name}.pth")
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "gradcam_maps", run_name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device)
    model = build_model(config, args.branch_setting, device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = make_dataset(args.dataset, config, args.split)
    cam_extractor = PatchGradCAM(model)
    last_index = min(args.index + args.count, len(dataset))

    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Saving Grad-CAM maps to: {output_dir}")

    try:
        for dataset_index in range(args.index, last_index):
            image_tensor, target = dataset[dataset_index]
            image = denormalize_image(image_tensor, config["mean"], config["std"])
            cam, pred, prob, class_id = cam_extractor(image_tensor.unsqueeze(0).to(device), args.target_class)
            heatmap = colorize_cam(cam, image.size)
            overlay = overlay_heatmap(image, heatmap, args.alpha)

            stem = f"{args.dataset}_{args.split}_{dataset_index:05d}"
            image.save(os.path.join(output_dir, f"{stem}_image.png"))
            heatmap.save(os.path.join(output_dir, f"{stem}_gradcam.png"))
            overlay.save(os.path.join(output_dir, f"{stem}_overlay.png"))
            np.savez_compressed(
                os.path.join(output_dir, f"{stem}_gradcam.npz"),
                gradcam=cam,
                target=np.array(target),
                prediction=np.array(pred),
                probability=np.array(prob),
                class_id=np.array(class_id),
            )
            print(f"index={dataset_index} target={target} pred={pred} prob={prob:.3f} cam_class={class_id}")
    finally:
        cam_extractor.remove()


if __name__ == "__main__":
    main()
