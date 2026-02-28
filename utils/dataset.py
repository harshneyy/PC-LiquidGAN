"""
utils/dataset.py
Dataset classes and preprocessing for PC-LiquidGAN.

Supported datasets:
  - ThermalDataset  : Generic paired RGB ↔ Thermal (KAIST, CBSR, Neonatal, FLIR)
  - UnpairedDataset : Unpaired images for zero-shot domain adaptation
"""

import os
import random
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  Paired Dataset (supervised training)
# ─────────────────────────────────────────────────────────────────────────────

class ThermalDataset(Dataset):
    """
    Paired RGB ↔ Thermal dataset.

    Directory structure expected:
        rgb_dir/       *.png / *.jpg  (visible spectrum images)
        thermal_dir/   *.png / *.jpg  (thermal / IR images, same filenames)

    Args:
        rgb_dir     (str): Path to RGB images folder.
        thermal_dir (str): Path to thermal images folder.
        img_size    (int): Resize both images to img_size × img_size.
        augment     (bool): Apply horizontal flip augmentation.
    """

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(
        self,
        rgb_dir:     str,
        thermal_dir: str,
        img_size:    int  = 256,
        augment:     bool = True,
    ):
        self.rgb_dir     = rgb_dir
        self.thermal_dir = thermal_dir
        self.augment     = augment

        # Gather files
        self.files = sorted([
            f for f in os.listdir(rgb_dir)
            if os.path.splitext(f)[1].lower() in self.IMG_EXTENSIONS
        ])

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No images found in {rgb_dir}. "
                f"Supported extensions: {self.IMG_EXTENSIONS}"
            )

        # Transforms
        self.rgb_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.thermal_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]

        rgb_img = Image.open(os.path.join(self.rgb_dir, fname)).convert('RGB')
        thermal_img = Image.open(os.path.join(self.thermal_dir, fname)).convert('L')

        # Consistent random horizontal flip
        if self.augment and random.random() > 0.5:
            rgb_img     = transforms.functional.hflip(rgb_img)
            thermal_img = transforms.functional.hflip(thermal_img)

        return self.rgb_transform(rgb_img), self.thermal_transform(thermal_img)


# ─────────────────────────────────────────────────────────────────────────────
#  Unpaired Dataset (zero-shot cross-domain evaluation)
# ─────────────────────────────────────────────────────────────────────────────

class UnpairedDataset(Dataset):
    """
    Unpaired image dataset for zero-shot cross-domain testing.
    Loads only RGB images (no ground-truth thermal required).

    Args:
        img_dir  (str): Path to RGB images folder.
        img_size (int): Resize images to img_size × img_size.
    """

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(self, img_dir: str, img_size: int = 256):
        self.img_dir = img_dir
        self.files   = sorted([
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in self.IMG_EXTENSIONS
        ])
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        fname = self.files[idx]
        img   = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        return self.transform(img), fname


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic test data generator (for quick testing without real data)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticThermalDataset(Dataset):
    """
    Generates random paired RGB-Thermal tensors for pipeline testing.
    Use this to verify model forward/backward passes before using real data.
    """

    def __init__(self, num_samples: int = 100, img_size: int = 64):
        self.num_samples = num_samples
        self.img_size    = img_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb     = torch.randn(3, self.img_size, self.img_size)
        thermal = torch.randn(1, self.img_size, self.img_size)
        return rgb, thermal
