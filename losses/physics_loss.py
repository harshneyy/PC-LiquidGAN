"""
losses/physics_loss.py
Physics-Informed Loss Functions for PC-LiquidGAN.

Implements:
  1. Heat Diffusion Loss  (L_flux):   dT/dt = α ∇²T
  2. Energy Conservation Loss (L_energy): ∫T dΩ ≈ constant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsLoss(nn.Module):
    """
    Combines heat diffusion and energy conservation constraints.

    Args:
        alpha (float): Thermal diffusivity constant α (default: 0.001)
    """

    def __init__(self, alpha: float = 0.001):
        super().__init__()
        self.alpha = alpha

        # Discrete 2D Laplacian kernel: approximates ∇²T
        lap = torch.tensor([[[[0.,  1., 0.],
                               [1., -4., 1.],
                               [0.,  1., 0.]]]])
        self.register_buffer('laplacian_kernel', lap)

    # ── Sub-losses ────────────────────────────────────────────────────────────

    def heat_diffusion_loss(
        self,
        T_pred: torch.Tensor,
        T_real: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalizes deviation from the heat equation: dT/dt ≈ α ∇²T
        
        * dT/dt  is approximated as (T_pred - T_real),
          assuming unit time between input (T_real) and output (T_pred).
        * ∇²T is computed via 2D discrete Laplacian convolution.
        """
        # Spatial Laplacian of predicted thermal field
        lap_T = F.conv2d(T_pred, self.laplacian_kernel, padding=1)
        # Temporal derivative approximation
        dT_dt = T_pred - T_real
        # Residual: dT/dt - α∇²T → should be 0
        return F.mse_loss(dT_dt, self.alpha * lap_T)

    def energy_conservation_loss(
        self,
        T_pred: torch.Tensor,
        T_real: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforces ∫T_pred dΩ ≈ ∫T_real dΩ (per-sample mean temperature).
        Prevents mode collapse to zero or saturated outputs.
        """
        E_pred = T_pred.mean(dim=[1, 2, 3])
        E_real = T_real.mean(dim=[1, 2, 3])
        return F.mse_loss(E_pred, E_real)

    def gradient_smoothness_loss(self, T_pred: torch.Tensor) -> torch.Tensor:
        """
        Optional: penalizes sharp spatial gradients in the thermal output.
        Encourages physically realistic smooth heat distributions.
        """
        # Horizontal and vertical finite differences
        dx = T_pred[:, :, :, 1:] - T_pred[:, :, :, :-1]
        dy = T_pred[:, :, 1:, :] - T_pred[:, :, :-1, :]
        return dx.abs().mean() + dy.abs().mean()

    # ── Combined forward ──────────────────────────────────────────────────────

    def forward(
        self,
        T_pred:         torch.Tensor,
        T_real:         torch.Tensor,
        lambda_flux:    float = 1.0,
        lambda_energy:  float = 0.5,
        lambda_smooth:  float = 0.1,
    ) -> torch.Tensor:
        """
        Args:
            T_pred        : Generated thermal image [B, 1, H, W]
            T_real        : Ground-truth thermal image [B, 1, H, W]
            lambda_flux   : Weight for heat diffusion loss
            lambda_energy : Weight for energy conservation loss
            lambda_smooth : Weight for gradient smoothness loss

        Returns:
            Total physics loss (scalar)
        """
        l_flux   = self.heat_diffusion_loss(T_pred, T_real)
        l_energy = self.energy_conservation_loss(T_pred, T_real)
        l_smooth = self.gradient_smoothness_loss(T_pred)

        return (lambda_flux   * l_flux   +
                lambda_energy * l_energy +
                lambda_smooth * l_smooth)
