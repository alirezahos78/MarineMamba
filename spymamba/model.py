import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm import Mamba


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
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


def get_spiral_indices(height, width, device="cpu"):
    coords = []
    top, bottom, left, right = 0, height - 1, 0, width - 1
    while top <= bottom and left <= right:
        for x in range(left, right + 1):
            coords.append(top * width + x)
        top += 1
        for y in range(top, bottom + 1):
            coords.append(y * width + right)
        right -= 1
        if top <= bottom:
            for x in range(right, left - 1, -1):
                coords.append(bottom * width + x)
            bottom -= 1
        if left <= right:
            for y in range(bottom, top - 1, -1):
                coords.append(y * width + left)
            left += 1
    idx = torch.tensor(coords, dtype=torch.long, device=device)
    return idx, torch.argsort(idx)


class SpiralScanner(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        idx_fwd, inv_fwd = get_spiral_indices(height, width)
        idx_bwd = torch.flip(idx_fwd, dims=[0])
        inv_bwd = torch.argsort(idx_bwd)
        self.register_buffer("idx_fwd", idx_fwd)
        self.register_buffer("inv_fwd", inv_fwd)
        self.register_buffer("idx_bwd", idx_bwd)
        self.register_buffer("inv_bwd", inv_bwd)

    def forward(self, x):
        return x[:, self.idx_fwd, :], x[:, self.idx_bwd, :]

    def inverse(self, fwd, bwd):
        return fwd[:, self.inv_fwd, :], bwd[:, self.inv_bwd, :]


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // ratio, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        mx  = self.fc2(self.relu(self.fc1(x.amax(dim=(-2, -1), keepdim=True))))
        return self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class SCAtt(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class SpyMambaBlock(nn.Module):
    """
    Bidirectional spiral Mamba block with CBAM attention, LayerScale, and DropPath.

    Accepts sequences of length H*W (spatial only) or 1+H*W (CLS + spatial).
    When a CLS token is present at position 0, it is prepended to the
    spiral-ordered spatial sequence before Mamba runs, so it attends to the
    full spatial context. After Mamba the CLS output is separated out and
    the spatial output is restored to raster order before CBAM is applied.
    """
    def __init__(self, dim, height, width, drop_path=0.0, ffn_drop=0.0, layer_scale_init=1.0):
        super().__init__()
        self.height = height
        self.width  = width
        self.num_spatial = height * width
        self.norm    = nn.LayerNorm(dim)
        self.scanner = SpiralScanner(height, width)
        self.mamba   = Mamba(d_model=dim, d_state=32, d_conv=4, expand=2)
        self.fusion_gate = nn.Parameter(torch.tensor(0.5))
        self.sc_att  = SCAtt(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(ffn_drop),
            nn.Linear(dim * 4, dim),
        )
        self.layer_scale_1 = nn.Parameter(torch.ones(dim) * layer_scale_init)
        self.layer_scale_2 = nn.Parameter(torch.ones(dim) * layer_scale_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # x: [B, L, dim]  where L = H*W  or  L = 1 + H*W
        B, L, C = x.shape
        has_cls = (L == self.num_spatial + 1)

        residual = x
        x = self.norm(x)

        if has_cls:
            cls_tok = x[:, :1, :]   # [B, 1, dim]
            spatial  = x[:, 1:, :]  # [B, H*W, dim]
        else:
            spatial = x

        # Spiral-reorder spatial, prepend CLS so it joins the Mamba sequence
        fwd_s, bwd_s = self.scanner(spatial)
        if has_cls:
            fwd = torch.cat([cls_tok, fwd_s], dim=1)  # [B, 1+H*W, dim]
            bwd = torch.cat([cls_tok, bwd_s], dim=1)
        else:
            fwd, bwd = fwd_s, bwd_s

        fwd = self.mamba(fwd)
        bwd = self.mamba(bwd)

        if has_cls:
            # Split CLS output and restore spatial to raster order
            fwd_cls, fwd_s = fwd[:, :1, :], fwd[:, 1:, :]
            bwd_cls, bwd_s = bwd[:, :1, :], bwd[:, 1:, :]
            y_cls = (fwd_cls + bwd_cls) * 0.5
            fwd_s, bwd_s = self.scanner.inverse(fwd_s, bwd_s)
        else:
            fwd, bwd = self.scanner.inverse(fwd, bwd)
            fwd_s, bwd_s = fwd, bwd

        y_spatial = self.fusion_gate * fwd_s + (1 - self.fusion_gate) * bwd_s

        # CBAM applied to spatial tokens only (they have 2D structure)
        y_2d = rearrange(y_spatial, "b (h w) c -> b c h w", h=self.height, w=self.width)
        y_2d = self.sc_att(y_2d)
        y_spatial = rearrange(y_2d, "b c h w -> b (h w) c")

        y = torch.cat([y_cls, y_spatial], dim=1) if has_cls else y_spatial

        x = residual + self.drop_path(self.layer_scale_1 * y)
        x = x + self.drop_path(self.layer_scale_2 * self.ffn(self.norm_ffn(x)))
        return x


class PyramidBranch(nn.Module):
    """
    Single-resolution branch.

    The CLIP CLS token is projected to dim and prepended to the spatial patch
    sequence before the Mamba blocks run:

        [cls_proj(CLS), patch_0, ..., patch_{H*W-1}]  →  SpyMambaBlocks  →  mean-pool

    Mean-pooling covers all 1+H*W tokens, so the output fuses global (CLS) and
    local (patch) information as shaped by the Mamba layers.
    """
    def __init__(self, input_dim, feature_h, feature_w, dim=128, depth=4,
                 cls_input_dim=512,
                 stochastic_depth=0.0, ffn_drop=0.0, layer_scale_init=1.0):
        super().__init__()
        self.feature_h = feature_h
        self.feature_w = feature_w
        num_spatial = feature_h * feature_w

        self.input_projection = nn.Linear(input_dim, dim)
        self.cls_projection   = nn.Linear(cls_input_dim, dim)
        # pos_embed covers [CLS position, patch_0, ..., patch_{H*W-1}]
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_spatial, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, depth)]
        self.blocks = nn.ModuleList([
            SpyMambaBlock(dim, feature_h, feature_w,
                          drop_path=dpr[i],
                          ffn_drop=ffn_drop,
                          layer_scale_init=layer_scale_init)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, cls):
        # x  : [B, C, H, W]  — spatial CLIP features
        # cls: [B, cls_input_dim]  — CLIP CLS token
        B = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(B, -1, x.shape[1])   # [B, H*W, C]
        x = self.input_projection(x)                              # [B, H*W, dim]

        cls_tok = self.cls_projection(cls).unsqueeze(1)           # [B, 1, dim]
        x = torch.cat([cls_tok, x], dim=1)                       # [B, 1+H*W, dim]
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x.mean(dim=1)                                      # [B, dim]


class PyramidCLIPSpyMamba(nn.Module):
    """
    Two parallel Mamba branches forming a feature pyramid.

    B/16 branch: [CLS_16, patch_0, ..., patch_195]  (14×14 = 196 spatial tokens)
    B/32 branch: [CLS_32, patch_0, ..., patch_48]   (7×7  =  49 spatial tokens)

    Each branch outputs a dim-d vector via mean-pooling over all tokens.
    The two outputs are concatenated and fed to an MLP head.
    """
    def __init__(
        self,
        fine_input_dim=768,
        fine_h=14,
        fine_w=14,
        coarse_input_dim=768,
        coarse_h=7,
        coarse_w=7,
        dim=128,
        fine_depth=1,
        coarse_depth=1,
        num_classes=20,
        cls_dim=512,
        stochastic_depth=0.0,
        ffn_drop=0.0,
        layer_scale_init=1.0,
        dropout=0.3,
        head_hidden_dim=512,
    ):
        super().__init__()
        self.fine_branch = PyramidBranch(
            fine_input_dim, fine_h, fine_w, dim, fine_depth,
            cls_input_dim=cls_dim,
            stochastic_depth=stochastic_depth,
            ffn_drop=ffn_drop,
            layer_scale_init=layer_scale_init,
        )
        self.coarse_branch = PyramidBranch(
            coarse_input_dim, coarse_h, coarse_w, dim, coarse_depth,
            cls_input_dim=cls_dim,
            stochastic_depth=stochastic_depth,
            ffn_drop=ffn_drop,
            layer_scale_init=layer_scale_init,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def get_features(self, fine_feats, coarse_feats, fine_cls, coarse_cls):
        """Pre-head representation [B, dim*2] for contrastive learning."""
        fine_out   = self.fine_branch(fine_feats,   fine_cls)
        coarse_out = self.coarse_branch(coarse_feats, coarse_cls)
        return torch.cat([fine_out, coarse_out], dim=-1)

    def forward(self, fine_feats, coarse_feats, fine_cls, coarse_cls):
        return self.head(self.get_features(fine_feats, coarse_feats, fine_cls, coarse_cls))
