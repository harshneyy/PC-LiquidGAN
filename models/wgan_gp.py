"""
models/wgan_gp.py
WGAN-GP Generator and Critic for baseline comparison.
Uses Wasserstein distance + Gradient Penalty instead of standard BCE adversarial loss.
No Neural ODE, no LNN, no physics loss.
"""

import torch
import torch.nn as nn


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


class WGANGenerator(nn.Module):
    """
    WGAN-GP encoder-decoder generator (same architecture as DCGAN for fair comparison).
    """
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            _enc_block(input_channels, 64, norm=False),
            _enc_block(64, 128),
            _enc_block(128, 256),
            _enc_block(256, 512),
            _enc_block(512, 512),
            _enc_block(512, 512),
        )
        self.decoder = nn.Sequential(
            _dec_block(512, 512),
            _dec_block(512, 512),
            _dec_block(512, 256),
            _dec_block(256, 128),
            _dec_block(128, 64),
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class WGANCritic(nn.Module):
    """
    WGAN-GP Critic (no sigmoid — outputs raw Wasserstein score).
    Uses InstanceNorm instead of BatchNorm for gradient penalty compatibility.
    """
    def __init__(self, input_channels=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            # No sigmoid for WGAN
        )

    def forward(self, rgb, thermal):
        x = torch.cat([rgb, thermal], dim=1)
        return self.model(x)


def compute_gradient_penalty(critic, rgb, real_thermal, fake_thermal, device):
    """Compute gradient penalty for WGAN-GP."""
    alpha = torch.rand(real_thermal.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real_thermal + (1 - alpha) * fake_thermal).requires_grad_(True)
    
    d_interpolated = critic(rgb, interpolated)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
