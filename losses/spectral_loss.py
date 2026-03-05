"""
losses/spectral_loss.py
Frequency-Domain Spectral Loss for PC-LiquidGAN.

Motivation:
    Real thermal images obey heat diffusion: ∂T/∂t = α∇²T.
    Solutions to this PDE are smooth — energy is concentrated in LOW
    spatial frequencies. A pixel-wise L1/L2 loss does not enforce this.

    We penalise the L2 distance between the FFT magnitude spectra of
    the generated and real thermal images, explicitly forcing the GAN
    to match the thermal frequency signature rather than just pixel values.

    L_spectral = || |FFT(T_pred)| - |FFT(T_real)| ||²

    To emphasise low-frequency accuracy we apply a radial Gaussian
    weight mask so that low-frequency errors are penalised MORE than
    high-frequency errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralLoss(nn.Module):
    """
    FFT-based frequency-domain loss for thermal image synthesis.

    Args:
        img_size  (int): Spatial size of the images (assumed square). Default 256.
        sigma     (float): Std-dev of the low-frequency Gaussian weight mask.
                           Smaller = emphasises lower frequencies more. Default 32.
        loss_type (str): 'mse' or 'mae'. Default 'mse'.
    """

    def __init__(self, img_size: int = 256, sigma: float = 32.0,
                 loss_type: str = 'mse'):
        super().__init__()
        self.loss_fn = F.mse_loss if loss_type == 'mse' else F.l1_loss

        # ── Build radial Gaussian weight mask (low-freq emphasis) ──────────────
        cy, cx = img_size // 2, img_size // 2
        ys = torch.arange(img_size).float() - cy
        xs = torch.arange(img_size).float() - cx
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        dist = torch.sqrt(yy ** 2 + xx ** 2)
        mask = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
        # Register as buffer so it moves to the correct device automatically
        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0))  # [1,1,H,W]

    def forward(self, T_pred: torch.Tensor,
                T_real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            T_pred: Generated thermal images  [B, 1, H, W]
            T_real: Ground-truth thermal images [B, 1, H, W]

        Returns:
            Weighted spectral loss (scalar)
        """
        # 2-D FFT → shift zero-frequency to centre → magnitude spectrum
        fft_pred = torch.fft.fft2(T_pred, norm='ortho')
        fft_real = torch.fft.fft2(T_real, norm='ortho')

        mag_pred = torch.fft.fftshift(fft_pred.abs())
        mag_real = torch.fft.fftshift(fft_real.abs())

        # Apply low-frequency emphasis mask
        weighted_pred = mag_pred * self.mask
        weighted_real = mag_real * self.mask

        return self.loss_fn(weighted_pred, weighted_real)
