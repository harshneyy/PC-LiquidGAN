"""
models/liquid_cell.py
Liquid Neural Network (LNN) Cell — the core building block of the discriminator.

The hidden state evolves via a continuous-time ODE:
    τ * dh/dt = -h + σ(W_in · x + W_rec · h + bias)
Discretized using Euler steps.
"""

import torch
import torch.nn as nn


class LiquidCell(nn.Module):
    """
    A single Liquid Neural Network cell.

    Args:
        input_size  (int): Dimension of the input vector.
        hidden_size (int): Dimension of the hidden state.
        num_steps   (int): Number of Euler integration steps (default: 6).
    """

    def __init__(self, input_size: int, hidden_size: int, num_steps: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_steps   = num_steps

        # Input projection
        self.W_in  = nn.Linear(input_size, hidden_size)
        # Recurrent projection (no bias to avoid double-counting)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        # Time constant τ — learnable, initialized to 1
        self.tau   = nn.Parameter(torch.ones(hidden_size))
        # Activation
        self.act   = nn.Tanh()

    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input  [batch, input_size]
            h (Tensor): Hidden [batch, hidden_size]  (None → zeros)
        Returns:
            h (Tensor): Updated hidden state [batch, hidden_size]
        """
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device)

        dt = 1.0 / self.num_steps
        for _ in range(self.num_steps):
            # Liquid ODE: dh/dt = (-h + act(W_in*x + W_rec*h)) / tau
            dh = (-h + self.act(self.W_in(x) + self.W_rec(h))) / (self.tau + 1e-6)
            h  = h + dt * dh

        return h
