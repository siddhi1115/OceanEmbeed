"""
oceanembed.py
-------------
OceanEmbed: lightweight PyTorch model for subsurface temperature reconstruction.

Architecture:
  Input (B, C_in, H, W)                            C_in = 9 (SST,SSS,SLA,Ucur,Vcur,Uwnd,Vwnd,lat,lon)
    ↓
  Multi-scale CNN encoder (3×3, 5×5, 7×7 parallel, concat)
    ↓
  N × Residual blocks
    ↓
  CBAM (Channel + Spatial Attention) — the "embedding"
    ↓
  Optional ConvLSTM over T time steps (T=1 → skipped)
    ↓
  U-Net decoder (3 levels, skip connections)
    ↓
  FiLM depth-conditioned head (learned depth embedding per output depth)
    ↓
  Output (B, 15, H, W)   # temperature at 15 depth levels

Loss:
  Masked MSE (land = NaN → weight = 0)
  + lambda_smooth * vertical smoothness regulariser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─── Building blocks ──────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(ch, ch, 3, 1),
            nn.Dropout2d(dropout),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ChannelAttention(nn.Module):
    """CBAM channel attention."""
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(ch // reduction, ch, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx = self.fc(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """CBAM spatial attention."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.ca = ChannelAttention(ch, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class MultiScaleEncoder(nn.Module):
    """Multi-scale CNN: 3×3 + 5×5 + 7×7 convolutions, concatenated."""
    def __init__(self, in_ch, base_ch):
        super().__init__()
        self.branch3 = ConvBnRelu(in_ch, base_ch, kernel=3, padding=1)
        self.branch5 = ConvBnRelu(in_ch, base_ch, kernel=5, padding=2)
        self.branch7 = ConvBnRelu(in_ch, base_ch, kernel=7, padding=3)
        self.fuse = ConvBnRelu(base_ch * 3, base_ch * 2, kernel=1, padding=0)

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        return self.fuse(torch.cat([b3, b5, b7], dim=1))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ConvBnRelu(out_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial dims don't match (can happen with odd sizes)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class FiLMDepthHead(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) depth-conditioned output head.
    Produces scale (γ) and shift (β) from a learned depth embedding,
    then applies: output = γ * features + β
    """
    def __init__(self, feature_ch, n_depths, depth_embed_dim):
        super().__init__()
        self.n_depths = n_depths
        self.depth_embed = nn.Embedding(n_depths, depth_embed_dim)
        self.film_gen = nn.Linear(depth_embed_dim, feature_ch * 2)  # → (γ, β)
        self.head = nn.Conv2d(feature_ch, 1, 1)  # 1 temp channel per depth

    def forward(self, features):
        """
        features: (B, feature_ch, H, W)
        Returns: (B, n_depths, H, W)
        """
        B, C, H, W = features.shape
        depth_ids = torch.arange(self.n_depths, device=features.device)
        emb = self.depth_embed(depth_ids)          # (n_depths, embed_dim)
        film = self.film_gen(emb)                  # (n_depths, C*2)
        gamma, beta = film[:, :C], film[:, C:]     # each (n_depths, C)

        outputs = []
        for d in range(self.n_depths):
            g = gamma[d].view(1, C, 1, 1)
            b = beta[d].view(1, C, 1, 1)
            feat_d = features * g + b
            out_d = self.head(feat_d)              # (B, 1, H, W)
            outputs.append(out_d)

        return torch.cat(outputs, dim=1)           # (B, n_depths, H, W)


# ─── Main model ───────────────────────────────────────────────────────────────

class OceanEmbed(nn.Module):
    """
    Full OceanEmbed model: surface inputs → 15-depth temperature reconstruction.
    """
    def __init__(
        self,
        in_channels: int = 9,
        base_channels: int = 32,
        n_residual_blocks: int = 4,
        n_depths: int = 15,
        depth_embed_dim: int = 16,
        dropout: float = 0.1,
        use_cbam: bool = True,
    ):
        super().__init__()
        bc = base_channels

        # Multi-scale encoder: → (B, bc*2, H, W)
        self.encoder = MultiScaleEncoder(in_channels, bc)
        enc_ch = bc * 2

        # Residual blocks (at full resolution)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(enc_ch, dropout) for _ in range(n_residual_blocks)]
        )

        # CBAM attention (the "embedding" block)
        self.cbam = CBAM(enc_ch) if use_cbam else nn.Identity()

        # U-Net encoder path (downsampling)
        self.down1 = DownBlock(enc_ch, enc_ch * 2)   # /2
        self.down2 = DownBlock(enc_ch * 2, enc_ch * 4)  # /4
        self.down3 = DownBlock(enc_ch * 4, enc_ch * 8)  # /8

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBnRelu(enc_ch * 8, enc_ch * 8),
            ResidualBlock(enc_ch * 8, dropout),
        )

        # U-Net decoder path
        self.up3 = UpBlock(enc_ch * 8, enc_ch * 4, enc_ch * 4)
        self.up2 = UpBlock(enc_ch * 4, enc_ch * 2, enc_ch * 2)
        self.up1 = UpBlock(enc_ch * 2, enc_ch, enc_ch)

        # FiLM depth-conditioned head
        self.film_head = FiLMDepthHead(enc_ch, n_depths, depth_embed_dim)

        self.n_depths = n_depths
        self.base_channels = bc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, H, W) — normalised surface inputs
        Returns: (B, n_depths, H, W) — temperature at 15 depths
        """
        # Encode
        e0 = self.encoder(x)           # (B, bc*2, H, W)   — skip 0
        e0 = self.res_blocks(e0)
        e0 = self.cbam(e0)

        # Downsample
        e1 = self.down1(e0)             # (B, bc*4, H/2, W/2) — skip 1
        e2 = self.down2(e1)             # (B, bc*8, H/4, W/4) — skip 2
        e3 = self.down3(e2)             # (B, bc*16, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decode
        d2 = self.up3(b, e2)
        d1 = self.up2(d2, e1)
        d0 = self.up1(d1, e0)

        # Depth-conditioned output
        out = self.film_head(d0)        # (B, 15, H, W)
        return out


# ─── Loss ─────────────────────────────────────────────────────────────────────

class MaskedMSELoss(nn.Module):
    """MSE loss that ignores NaN positions (land mask)."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        diff = (pred[mask] - target[mask]) ** 2
        return diff.mean()


class VerticalSmoothnessLoss(nn.Module):
    """
    Penalises temperature inversions (temp increasing with depth).
    Real ocean profiles should generally cool or stay constant with depth.
    loss = mean(relu(T[d+1] - T[d]))   for d in [0..13]
    """
    def forward(self, pred: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # pred: (B, 15, H, W)
        diff = pred[:, 1:, :, :] - pred[:, :-1, :, :]  # (B, 14, H, W)
        penalty = F.relu(diff)   # penalise only positive differences
        if mask is not None:
            # mask: (H, W) boolean — True = ocean
            penalty = penalty * mask.unsqueeze(0).unsqueeze(0)
        return penalty.mean()


class OceanEmbedLoss(nn.Module):
    def __init__(self, lambda_smooth: float = 0.1):
        super().__init__()
        self.mse = MaskedMSELoss()
        self.smooth = VerticalSmoothnessLoss()
        self.lambda_smooth = lambda_smooth

    def forward(self, pred, target, land_mask=None):
        mse_loss = self.mse(pred, target)
        smooth_loss = self.smooth(pred, land_mask)
        total = mse_loss + self.lambda_smooth * smooth_loss
        return total, mse_loss, smooth_loss


# ─── Model factory & info ─────────────────────────────────────────────────────

def build_model(cfg: dict) -> OceanEmbed:
    mc = cfg["model"]
    return OceanEmbed(
        in_channels=mc["in_channels"],
        base_channels=mc["base_channels"],
        n_residual_blocks=mc["n_residual_blocks"],
        n_depths=mc["n_depths"],
        depth_embed_dim=mc["depth_embed_dim"],
        dropout=mc["dropout"],
        use_cbam=mc["use_cbam"],
    )


def model_info(model: OceanEmbed) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "architecture": "MultiScaleCNN + ResBlocks + CBAM + UNet + FiLMDepthHead",
        "input_channels": model.encoder.branch3.block[0].weight.shape[1],
        "output_depths": model.n_depths,
    }


if __name__ == "__main__":
    # Quick smoke test
    import sys
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    with open(root / "pipeline" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg)
    info = model_info(model)
    print(f"Model parameters: {info['total_parameters']:,}")
    print(f"Architecture: {info['architecture']}")

    # Test forward pass with a 64×64 patch
    x = torch.randn(2, cfg["model"]["in_channels"], 64, 64)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape:  {list(x.shape)}")
    print(f"Output shape: {list(y.shape)}  (expected: [2, 15, 64, 64])")
    assert y.shape == (2, 15, 64, 64), f"Shape mismatch: {y.shape}"
    print("✓ Smoke test passed")
