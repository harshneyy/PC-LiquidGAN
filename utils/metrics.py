"""
utils/metrics.py
Evaluation metrics for PC-LiquidGAN:
  - SSIM  (Structural Similarity Index)
  - PSNR  (Peak Signal-to-Noise Ratio)
  - FID   (Fréchet Inception Distance) — via torch-fidelity
  - RMSE  (Root Mean Square Error)
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a [-1,1] normalized tensor to [0,1] numpy array."""
    img = t.detach().cpu().numpy()
    img = (img + 1.0) / 2.0      # [-1,1] → [0,1]
    img = np.clip(img, 0.0, 1.0)
    return img


def compute_ssim(pred: torch.Tensor, real: torch.Tensor) -> float:
    """
    Compute mean SSIM over a batch.
    
    Args:
        pred (Tensor): [B, 1, H, W] generated thermal
        real (Tensor): [B, 1, H, W] ground-truth thermal
    Returns:
        mean SSIM (float)
    """
    pred_np = tensor_to_numpy(pred)  # [B, 1, H, W]
    real_np = tensor_to_numpy(real)
    scores = []
    for i in range(pred_np.shape[0]):
        p = pred_np[i, 0]  # [H, W]
        r = real_np[i, 0]
        scores.append(ssim_fn(r, p, data_range=1.0))
    return float(np.mean(scores))


def compute_psnr(pred: torch.Tensor, real: torch.Tensor) -> float:
    """
    Compute mean PSNR over a batch.

    Args:
        pred (Tensor): [B, 1, H, W] generated thermal
        real (Tensor): [B, 1, H, W] ground-truth thermal
    Returns:
        mean PSNR in dB (float)
    """
    pred_np = tensor_to_numpy(pred)
    real_np = tensor_to_numpy(real)
    scores = []
    for i in range(pred_np.shape[0]):
        p = pred_np[i, 0]
        r = real_np[i, 0]
        scores.append(psnr_fn(r, p, data_range=1.0))
    return float(np.mean(scores))


def compute_rmse(pred: torch.Tensor, real: torch.Tensor) -> float:
    """
    Compute RMSE in [0,1] pixel space.
    """
    pred_np = tensor_to_numpy(pred)
    real_np = tensor_to_numpy(real)
    return float(np.sqrt(np.mean((pred_np - real_np) ** 2)))
