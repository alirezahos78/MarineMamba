import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm import Mamba


def get_branch_flags(setting_name):
    branch_flags = {
        "use_local_2": True,
        "use_local_4": True,
        "use_local_8": True,
    }
    if setting_name == "full":
        return branch_flags
    if setting_name == "no_local_2":
        branch_flags["use_local_2"] = False
        return branch_flags
    if setting_name == "no_local_8":
        branch_flags["use_local_8"] = False
        return branch_flags
    raise ValueError(f"Unknown branch ablation setting: {setting_name}")


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def get_spiral_indices_out_in(height, width, device="cpu"):
    coords = []
    top, bottom = 0, height - 1
    left, right = 0, width - 1

    while top <= bottom and left <= right:
        for x_pos in range(left, right + 1):
            coords.append(top * width + x_pos)
        top += 1
        for y_pos in range(top, bottom + 1):
            coords.append(y_pos * width + right)
        right -= 1
        if top <= bottom:
            for x_pos in range(right, left - 1, -1):
                coords.append(bottom * width + x_pos)
            bottom -= 1
        if left <= right:
            for y_pos in range(bottom, top - 1, -1):
                coords.append(y_pos * width + left)
            left += 1

    spiral_indices = torch.tensor(coords, dtype=torch.long, device=device)
    return spiral_indices, torch.argsort(spiral_indices)


class UniversalScanner(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        idx_fwd, inv_fwd = get_spiral_indices_out_in(height, width)
        idx_bwd = torch.flip(idx_fwd, dims=[0])
        inv_bwd = torch.argsort(idx_bwd)

        self.register_buffer("idx_fwd", idx_fwd)
        self.register_buffer("inv_fwd", inv_fwd)
        self.register_buffer("idx_bwd", idx_bwd)
        self.register_buffer("inv_bwd", inv_bwd)

    def scan_bidirectional(self, x):
        return x[:, self.idx_fwd, :], x[:, self.idx_bwd, :]

    def unscan_bidirectional(self, fwd, bwd):
        return fwd[:, self.inv_fwd, :], bwd[:, self.inv_bwd, :]


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class SCAtt(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class UniversalVimBlock(nn.Module):
    def __init__(self, dim, height, width, drop_path=0.0, block_id=0, branch_flags=None):
        super().__init__()
        self.height = height
        self.width = width
        self.block_id = block_id
        self.norm = nn.LayerNorm(dim)
        branch_flags = branch_flags or {}
        self.use_local_2 = branch_flags.get("use_local_2", True)
        self.use_local_4 = branch_flags.get("use_local_4", True)
        self.use_local_8 = branch_flags.get("use_local_8", True)

        self.global_scanner = UniversalScanner(height, width)
        self.mamba = Mamba(d_model=dim, d_state=32, d_conv=4, expand=2)
        self.fusion_gate = nn.Parameter(torch.tensor(0.5))

        self.h_local_2 = height // 2
        self.w_local_2 = width // 2
        if self.use_local_2 and self.h_local_2 > 0 and self.w_local_2 > 0:
            self.local_scanner_2 = UniversalScanner(self.h_local_2, self.w_local_2)
            self.local_mamba_2 = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
            self.fusion_gate_2 = nn.Parameter(torch.tensor(0.5))
            self.learnable_lambda_2 = nn.Parameter(torch.tensor(0.1))
        else:
            self.local_scanner_2 = None

        self.h_local_4 = max(1, height // 4)
        self.w_local_4 = max(1, width // 4)
        if self.use_local_4:
            self.local_scanner_4 = UniversalScanner(self.h_local_4, self.w_local_4)
            self.local_mamba_4 = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
            self.fusion_gate_4 = nn.Parameter(torch.tensor(0.5))
            self.learnable_lambda_4 = nn.Parameter(torch.tensor(0.1))
        else:
            self.local_scanner_4 = None

        self.h_local_8 = max(1, height // 8)
        self.w_local_8 = max(1, width // 8)
        if self.use_local_8:
            self.local_scanner_8 = UniversalScanner(self.h_local_8, self.w_local_8)
            self.local_mamba_8 = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
            self.fusion_gate_8 = nn.Parameter(torch.tensor(0.5))
            self.learnable_lambda_8 = nn.Parameter(torch.tensor(0.1))
        else:
            self.local_scanner_8 = None

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.sc_att = SCAtt(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def process_bidirectional(self, x, scanner, mamba, fusion_gate):
        fwd_seq, bwd_seq = scanner.scan_bidirectional(x)
        fwd_processed = mamba(fwd_seq)
        bwd_processed = mamba(bwd_seq)
        fwd_unscan, bwd_unscan = scanner.unscan_bidirectional(fwd_processed, bwd_processed)
        return fusion_gate * fwd_unscan + (1 - fusion_gate) * bwd_unscan

    def forward_local(self, x, h_local, w_local, scanner, mamba, fusion_gate):
        x_2d = rearrange(x, "b (h w) c -> b c h w", h=self.height, w=self.width)
        x_down = torch.nn.functional.adaptive_avg_pool2d(x_2d, (h_local, w_local))
        x_down_flat = rearrange(x_down, "b c h w -> b (h w) c")
        y_down = self.process_bidirectional(x_down_flat, scanner, mamba, fusion_gate)
        y_down_2d = rearrange(y_down, "b (h w) c -> b c h w", h=h_local, w=w_local)
        y_up = torch.nn.functional.interpolate(
            y_down_2d,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        return rearrange(y_up, "b c h w -> b (h w) c")

    def forward(self, x, return_lambdas=False):
        residual = x
        x = self.norm(x)
        y_combined = self.process_bidirectional(x, self.global_scanner, self.mamba, self.fusion_gate)
        lambdas_dict = {}

        if self.local_scanner_2 is not None:
            y_local_2 = self.forward_local(
                x, self.h_local_2, self.w_local_2, self.local_scanner_2, self.local_mamba_2, self.fusion_gate_2
            )
            y_combined = y_combined + self.learnable_lambda_2 * y_local_2
            lambdas_dict["lambda_2"] = self.learnable_lambda_2.item()

        if self.local_scanner_4 is not None:
            y_local_4 = self.forward_local(
                x, self.h_local_4, self.w_local_4, self.local_scanner_4, self.local_mamba_4, self.fusion_gate_4
            )
            y_combined = y_combined + self.learnable_lambda_4 * y_local_4
            lambdas_dict["lambda_4"] = self.learnable_lambda_4.item()

        if self.local_scanner_8 is not None:
            y_local_8 = self.forward_local(
                x, self.h_local_8, self.w_local_8, self.local_scanner_8, self.local_mamba_8, self.fusion_gate_8
            )
            y_combined = y_combined + self.learnable_lambda_8 * y_local_8
            lambdas_dict["lambda_8"] = self.learnable_lambda_8.item()

        y_combined_2d = rearrange(y_combined, "b (h w) c -> b c h w", h=self.height, w=self.width)
        y_att = self.sc_att(y_combined_2d)
        y_combined = rearrange(y_att, "b c h w -> b (h w) c")

        x = residual + self.drop_path(y_combined)
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        if return_lambdas:
            return x, lambdas_dict
        return x


class UniversalMambaTinyImageNet(nn.Module):
    def __init__(
        self,
        patch_size=2,
        dim=128,
        depth=6,
        num_classes=200,
        img_size=32,
        in_channels=3,
        stochastic_depth=0.0,
        branch_flags=None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.depth = depth
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")
        self.h_feat = img_size // patch_size
        self.w_feat = img_size // patch_size
        num_patches = self.h_feat * self.w_feat

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, depth)]
        self.blocks = nn.ModuleList([
            UniversalVimBlock(
                dim,
                self.h_feat,
                self.w_feat,
                drop_path=dpr[i],
                block_id=i,
                branch_flags=branch_flags,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(dim, num_classes))

    def forward_features(self, x, return_lambdas=False):
        x = self.patch_embed(x)
        x = x.transpose(1, 2)
        x = x + self.pos_embed

        all_lambdas = []
        for block in self.blocks:
            if return_lambdas:
                x, lambdas = block(x, return_lambdas=True)
                all_lambdas.append(lambdas)
            else:
                x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        if return_lambdas:
            return x, all_lambdas
        return x

    def forward(self, x, return_lambdas=False):
        if return_lambdas:
            features, all_lambdas = self.forward_features(x, return_lambdas=True)
            return self.head(features), all_lambdas
        return self.head(self.forward_features(x))


class SpatialFeatureMambaClassifier(nn.Module):
    def __init__(
        self,
        input_dim=768,
        feature_height=7,
        feature_width=7,
        patch_size=1,
        dim=128,
        depth=6,
        num_classes=20,
        stochastic_depth=0.0,
        branch_flags=None,
        dropout=0.15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.patch_size = patch_size
        self.depth = depth
        if feature_height % patch_size != 0 or feature_width % patch_size != 0:
            raise ValueError(
                f"feature grid ({feature_height}x{feature_width}) must be divisible by patch_size ({patch_size})"
            )
        self.h_feat = feature_height // patch_size
        self.w_feat = feature_width // patch_size
        num_patches = self.h_feat * self.w_feat

        if patch_size == 1:
            self.input_projection = nn.Linear(input_dim, dim)
        else:
            self.input_projection = nn.Conv2d(input_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, depth)]
        self.blocks = nn.ModuleList([
            UniversalVimBlock(
                dim,
                self.h_feat,
                self.w_feat,
                drop_path=dpr[i],
                block_id=i,
                branch_flags=branch_flags,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes))

    def forward_features(self, x, return_lambdas=False):
        if x.ndim == 4 and self.patch_size > 1:
            x = self.input_projection(x)
            x = rearrange(x, "b c h w -> b (h w) c")
        elif x.ndim == 4:
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.input_projection(x)
        elif x.ndim == 3 and self.patch_size > 1:
            x = rearrange(x, "b (h w) c -> b c h w", h=self.feature_height, w=self.feature_width)
            x = self.input_projection(x)
            x = rearrange(x, "b c h w -> b (h w) c")
        elif x.ndim == 3:
            x = self.input_projection(x)
        if x.ndim != 3:
            raise ValueError(f"Expected [B, C, H, W] or [B, N, C] features, got shape {tuple(x.shape)}")
        x = x + self.pos_embed

        all_lambdas = []
        for block in self.blocks:
            if return_lambdas:
                x, lambdas = block(x, return_lambdas=True)
                all_lambdas.append(lambdas)
            else:
                x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        if return_lambdas:
            return x, all_lambdas
        return x

    def forward(self, x, return_lambdas=False):
        if return_lambdas:
            features, all_lambdas = self.forward_features(x, return_lambdas=True)
            return self.head(features), all_lambdas
        return self.head(self.forward_features(x))
