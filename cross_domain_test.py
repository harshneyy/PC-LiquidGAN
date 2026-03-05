import torch
import os
from torch.utils.data import DataLoader
from models.generator import NeuralODEGenerator
from utils.dataset import ThermalDataset
from utils.metrics import compute_ssim, compute_psnr
from config import Config
import argparse

def test_cross_domain(model_path, dataset_name):
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    G = NeuralODEGenerator(
        input_channels=3,
        output_channels=1,
        latent_dim=cfg.LATENT_DIM,
        ode_method=cfg.ODE_METHOD
    ).to(device)
    G.load_state_dict(checkpoint['G_state'])
    G.eval()
    
    # Load dataset
    base_path = os.path.join(cfg.DATA_DIR, dataset_name)
    val_rgb_path = os.path.join(base_path, 'val', 'rgb')
    val_thr_path = os.path.join(base_path, 'val', 'thermal')
    
    # fallback to flat if needed
    if not os.path.exists(val_rgb_path):
        val_rgb_path = os.path.join(base_path, 'rgb')
        val_thr_path = os.path.join(base_path, 'thermal')
    
    dataset = ThermalDataset(val_rgb_path, val_thr_path, img_size=cfg.IMG_SIZE)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    total_ssim = 0
    total_psnr = 0
    n = len(loader)
    
    print(f"\nTesting {os.path.basename(model_path)} on {dataset_name} dataset...")
    
    with torch.no_grad():
        for i, (rgb, real) in enumerate(loader):
            rgb = rgb.to(device)
            real = real.to(device)
            fake = G(rgb)
            
            total_ssim += compute_ssim(fake, real)
            total_psnr += compute_psnr(fake, real)
            
            if i == 0:
                # Save first batch for visual inspection
                from torchvision.utils import save_image
                # Repeat 1-channel thermal to 3-channels for grid with RGB
                fake_rgb = fake.repeat(1, 3, 1, 1)
                real_rgb = real.repeat(1, 3, 1, 1)
                grid = torch.cat([rgb, fake_rgb, real_rgb], dim=0)
                os.makedirs('results/cross_domain', exist_ok=True)
                save_image(grid * 0.5 + 0.5, f'results/cross_domain/{dataset_name}_test.png', nrow=8)

    print(f"Final Cross-Domain Results:")
    print(f"  SSIM: {total_ssim/n:.4f}")
    print(f"  PSNR: {total_psnr/n:.2f} dB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    test_cross_domain(args.model, args.dataset)
