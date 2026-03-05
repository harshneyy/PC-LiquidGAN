"""
models/dcgan_generator.py
Standard DCGAN Generator for baseline comparison.
Maps RGB image → Thermal image using a simple encoder-decoder CNN (no Neural ODE, no physics).
"""

import torch
import torch.nn as nn


def _enc_block(in_ch, out_ch, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _dec_block(in_ch, out_ch, norm=True):
    layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DCGANGenerator(nn.Module):
    """
    Standard DCGAN-style encoder-decoder generator.
    No Neural ODE, no physics loss — purely data-driven.
    """

    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()

        # Encoder: [B, 3, 256, 256] → [B, 512, 4, 4]
        self.encoder = nn.Sequential(
            _enc_block(input_channels, 64, norm=False),  # 128
            _enc_block(64, 128),                          # 64
            _enc_block(128, 256),                         # 32
            _enc_block(256, 512),                         # 16
            _enc_block(512, 512),                         # 8
            _enc_block(512, 512),                         # 4
        )

        # Decoder: [B, 512, 4, 4] → [B, 1, 256, 256]
        self.decoder = nn.Sequential(
            _dec_block(512, 512),                         # 8
            _dec_block(512, 512),                         # 16
            _dec_block(512, 256),                         # 32
            _dec_block(256, 128),                         # 64
            _dec_block(128, 64),                          # 128
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1),  # 256
            nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DCGANDiscriminator(nn.Module):
    """
    Standard DCGAN PatchGAN discriminator.
    No Liquid Neural Network — uses fixed convolutional filters.
    """

    def __init__(self, input_channels=4):
        """input_channels = 3 (RGB) + 1 (thermal) = 4"""
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, rgb, thermal):
        x = torch.cat([rgb, thermal], dim=1)
        return self.model(x)
