"""
models/discriminator.py
LNN-based Discriminator for PC-LiquidGAN.

Architecture:
  1. CNN backbone  → extract spatial features
  2. Flatten
  3. LiquidCell    → dynamic temporal feature refinement
  4. FC            → real/fake score
"""

import torch
import torch.nn as nn
from models.liquid_cell import LiquidCell


def _conv_block(in_ch: int, out_ch: int, norm: bool = True) -> nn.Sequential:
    """Helper: Conv2d → (BatchNorm) → LeakyReLU block."""
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class LiquidDiscriminator(nn.Module):
    """
    PatchGAN-style CNN + Liquid Neural Network discriminator.

    Args:
        img_channels (int): Number of input image channels (1 for thermal).
        hidden_size  (int): LNN hidden state dimension.
    """

    def __init__(self, img_channels: int = 1, hidden_size: int = 256):
        super().__init__()

        # CNN: 256→128→64→32→16, channels: 1→64→128→256→512
        self.cnn = nn.Sequential(
            _conv_block(img_channels, 64,  norm=False),  # 128×128
            _conv_block(64,  128),                        # 64×64
            _conv_block(128, 256),                        # 32×32
            _conv_block(256, 512),                        # 16×16
            nn.AdaptiveAvgPool2d((4, 4)),                 # 4×4  (feat: 512×4×4)
        )
        feat_dim = 512 * 4 * 4   # = 8192

        # Liquid Neural Network refinement
        self.liquid = LiquidCell(input_size=feat_dim, hidden_size=hidden_size)

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Image [B, C, H, W]
        Returns:
            score (Tensor): Real/fake probability [B, 1]
        """
        feats  = self.cnn(x)                           # [B, 512, 4, 4]
        feats  = feats.view(feats.size(0), -1)         # [B, 8192]
        h      = self.liquid(feats)                    # [B, hidden_size]
        return self.classifier(h)                      # [B, 1]
