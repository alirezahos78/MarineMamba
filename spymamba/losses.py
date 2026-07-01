import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.reduction       = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits, targets):
        num_classes = logits.size(1)

        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_p = F.log_softmax(logits, dim=1)
            ce = -(smooth_targets * log_p).sum(dim=1)
            p_t = torch.exp(-ce)
        else:
            log_p = F.log_softmax(logits, dim=1)
            ce  = F.nll_loss(log_p, targets, reduction="none")
            p_t = torch.exp(-ce)

        focal_weight = (1 - p_t) ** self.gamma
        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha[targets]

        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(config, num_classes, class_counts=None, device="cpu"):
    loss_type = config.get("loss", "cross_entropy")
    ls        = config.get("label_smoothing", 0.0)

    if loss_type == "focal":
        gamma = config.get("focal_gamma", 2.0)
        alpha = None
        if config.get("focal_alpha") == "balanced" and class_counts is not None:
            counts = class_counts.float().to(device)
            alpha  = (counts.sum() / (num_classes * counts)).clamp(min=1e-6)
            alpha  = alpha / alpha.sum() * num_classes
        criterion = FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=ls).to(device)
        print(f"   Loss: FocalLoss  gamma={gamma}  "
              f"alpha={'balanced' if alpha is not None else 'uniform'}  ls={ls}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)
        print(f"   Loss: CrossEntropyLoss  ls={ls}")

    return criterion
