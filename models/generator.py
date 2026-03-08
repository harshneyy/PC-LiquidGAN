"""
models/generator.py
Neural ODE Generator for PC-LiquidGAN.

Architecture:
  Encoder (CNN)  →  z0 (latent)  →  Neural ODE evolves z0→z1  →  Decoder (CNN)  →  Thermal image

The Neural ODE models continuous-time latent dynamics:
    dz/dt = f_θ(t, z)   solved from t=0 to t=1
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


# ─────────────────────────────────────────────────────────────────────────────
#  ODE Dynamics Function
# ─────────────────────────────────────────────────────────────────────────────

class ODEFunc(nn.Module):
    """
    Parameterizes the right-hand side of dz/dt = f_θ(t, z).
    Uses a residual MLP with time embedding.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.nfe = 0  # Number of function evaluations (for monitoring)

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),   # +1 for time t
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )

        # Initialize last layer near zero for stability
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        # Append time as an extra feature
        t_vec = t.expand(z.size(0), 1)
        return self.net(torch.cat([z, t_vec], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
#  Encoder / Decoder helpers
# ─────────────────────────────────────────────────────────────────────────────

def _enc_block(in_ch, out_ch, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _dec_block(in_ch, out_ch, norm=True):
    layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
#  Neural ODE Generator
# ─────────────────────────────────────────────────────────────────────────────

class NeuralODEGenerator(nn.Module):
    """
    Generator: RGB image → Thermal image via Neural ODE latent dynamics.

    Args:
        input_channels  (int): Channels in input RGB  (default: 3)
        output_channels (int): Channels in output thermal (default: 1)
        latent_dim      (int): Latent ODE state dimension (default: 128)
        ode_method      (str): ODE solver (default: 'dopri5')
        rtol, atol      (float): Solver tolerances
    """

    def __init__(
        self,
        input_channels:  int   = 3,
        output_channels: int   = 1,
        latent_dim:      int   = 128,
        ode_method:      str   = 'dopri5',
        rtol:            float = 1e-3,
        atol:            float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.ode_method = ode_method
        self.rtol       = rtol
        self.atol       = atol

        # ── Encoder: RGB [B,3,256,256] → feature map [B,256,16,16] ──────────
        self.encoder = nn.Sequential(
            _enc_block(input_channels, 64,  norm=False),  # 128×128
            _enc_block(64,  128),                          # 64×64
            _enc_block(128, 256),                          # 32×32
            _enc_block(256, 512),                          # 16×16
        )
        # Project to latent vector
        self.enc_pool = nn.AdaptiveAvgPool2d((4, 4))       # → [B,512,4,4]
        self.enc_fc   = nn.Linear(512 * 4 * 4, latent_dim)

        # ── Neural ODE ───────────────────────────────────────────────────────
        self.ode_func = ODEFunc(latent_dim)
        self.register_buffer('t_span', torch.tensor([0.0, 1.0]))

        # ── Decoder: latent → thermal [B,1,256,256] ──────────────────────────
        self.dec_fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            _dec_block(512, 256),                          # 8×8
            _dec_block(256, 128),                          # 16×16
            _dec_block(128, 64),                           # 32×32
            _dec_block(64,  32),                           # 64×64
            _dec_block(32,  16),                           # 128×128
            nn.ConvTranspose2d(16, output_channels, 4, 2, 1),  # 256×256
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): RGB image [B, 3, H, W]
        Returns:
            out (Tensor): Synthetic thermal image [B, 1, H, W]
        """
        B = x.size(0)

        # Encode
        enc  = self.encoder(x)           # [B, 512, 16, 16]
        enc  = self.enc_pool(enc)         # [B, 512, 4, 4]
        enc  = enc.view(B, -1)            # [B, 8192]
        z0   = self.enc_fc(enc)           # [B, latent_dim]

        # Neural ODE: evolve z0 from t=0 to t=1
        t = self.t_span.to(x.device)
        z_t = odeint(
            self.ode_func, z0, t,
            method=self.ode_method,
            rtol=self.rtol, atol=self.atol,
            adjoint_params=list(self.ode_func.parameters())
        )
        z1 = z_t[-1]                      # [B, latent_dim] at t=1

        # Decode
        dec = self.dec_fc(z1)             # [B, 512*4*4]
        dec = dec.view(B, 512, 4, 4)      # [B, 512, 4, 4]
        return self.decoder(dec)           # [B, 1, 256, 256]


# ─────────────────────────────────────────────────────────────────────────────
#  Neural ODE UNet Generator (High Fidelity)
# ─────────────────────────────────────────────────────────────────────────────

class ConvODEFunc(nn.Module):
    """Spatial ODE function for 2D feature maps."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim + 1, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H, W]
        t_vec = t.view(1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))
        return self.net(torch.cat([x, t_vec], dim=1))


class ODEUNetGenerator(nn.Module):
    """
    UNet Generator with Neural ODE Bottleneck.
    Skip connections preserve high-frequency spatial details (sharpness).
    """
    def __init__(self, input_channels=3, output_channels=1, ode_method='euler'):
        super().__init__()
        self.ode_method = ode_method  # 'euler' or 'rk4' recommended for speed on Conv2D
        self.rtol = 1e-3
        self.atol = 1e-3

        # Encoder
        self.e1 = _enc_block(input_channels, 64, norm=False) # 128x128
        self.e2 = _enc_block(64, 128)                        # 64x64
        self.e3 = _enc_block(128, 256)                       # 32x32
        self.e4 = _enc_block(256, 512)                       # 16x16

        # Spatial Bottleneck + ConvODE
        self.bottleneck_in = nn.Conv2d(512, 64, 1)           # Compress channels for fast ODE
        self.ode_func = ConvODEFunc(64)                      # ODE operates on 64 channels
        self.register_buffer('t_span', torch.tensor([0.0, 1.0]))
        self.bottleneck_out = nn.Conv2d(64, 512, 1)

        # Decoder (with skip connections)
        self.d4 = _dec_block(512 + 512, 256) # 32x32
        self.d3 = _dec_block(256 + 256, 128) # 64x64
        self.d2 = _dec_block(128 + 128, 64)  # 128x128
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, 4, 2, 1), # 256x256
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, output_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsample
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        # ODE Bottleneck
        z0 = self.bottleneck_in(e4)
        t  = self.t_span.to(x.device)
        z_t = odeint(
            self.ode_func, z0, t,
            method=self.ode_method,
            rtol=self.rtol, atol=self.atol,
            adjoint_params=list(self.ode_func.parameters())
        )
        z1 = z_t[-1]
        b_out = self.bottleneck_out(z1)

        # Upsample with Skip Connections
        d4 = self.d4(torch.cat([b_out, e4], dim=1))
        d3 = self.d3(torch.cat([d4, e3], dim=1))
        d2 = self.d2(torch.cat([d3, e2], dim=1))
        out = self.d1(torch.cat([d2, e1], dim=1))
        return out
