"""
losses/perceptual_loss.py
VGG16 Feature-Matching Perceptual Loss for PC-LiquidGAN.

Why this fixes the washed-out output problem:
    L1 pixel loss: penalises each pixel independently → model learns
                   the MEAN of all plausible outputs → blurry, grey.

    VGG perceptual loss: compares FEATURE MAPS from a pre-trained
    VGG16 at multiple depths (relu1_2, relu2_2, relu3_3).
    These features encode edges, textures, and structural patterns.
    The model is forced to reproduce high-frequency detail to match
    these features — producing sharp, high-contrast thermal outputs.

Reference: Johnson et al., "Perceptual Losses for Real-Time Style
Transfer and Super-Resolution", ECCV 2016.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGGFeatureExtractor(nn.Module):
    """Extract multi-scale features from VGG16 (frozen)."""

    def __init__(self, layers=(3, 8, 15)):
        """
        Args:
            layers: VGG16 feature layer indices to extract from.
                    Default (3, 8, 15) = relu1_2, relu2_2, relu3_3.
        """
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.layers = layers
        self.blocks = nn.ModuleList()

        prev = 0
        for idx in layers:
            self.blocks.append(nn.Sequential(*list(vgg.features[prev:idx + 1])))
            prev = idx + 1

        # Freeze weights — we use VGG only as a fixed feature extractor
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class PerceptualLoss(nn.Module):
    """
    Multi-scale VGG perceptual loss.

    Handles grayscale (1-ch) thermal images by repeating to 3 channels
    before passing through VGG.

    Args:
        weights (tuple): Per-scale loss weights. Default: (1, 0.5, 0.25)
                         (emphasise shallow features which encode sharp edges)
    """

    def __init__(self, weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.vgg   = VGGFeatureExtractor()
        self.weights = weights
        self.criterion = nn.L1Loss()

        # VGG ImageNet normalization
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert grayscale [B,1,H,W] in [-1,1] to VGG-ready [B,3,H,W].
        """
        # De-normalise from [-1,1] to [0,1]
        x = x * 0.5 + 0.5
        # Grayscale → RGB by repeating channel
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # VGG ImageNet normalisation
        x = (x - self.mean) / self.std
        return x

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   Generated thermal image  [B, 1, H, W]
            target: Real thermal image       [B, 1, H, W]

        Returns:
            Weighted perceptual loss (scalar)
        """
        pred_vgg   = self._prepare(pred)
        target_vgg = self._prepare(target.detach())  # detach target from graph

        pred_feats   = self.vgg(pred_vgg)
        target_feats = self.vgg(target_vgg)

        loss = 0.0
        for w, pf, tf in zip(self.weights, pred_feats, target_feats):
            loss = loss + w * self.criterion(pf, tf)
        return loss
