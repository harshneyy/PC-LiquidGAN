"""
test_pipeline.py - Quick sanity check for PC-LiquidGAN
Run:  python test_pipeline.py
"""
import torch
import torch.nn as nn

print("Testing PC-LiquidGAN pipeline...\n")

print("[1/5] Testing LiquidCell...")
from models.liquid_cell import LiquidCell
cell = LiquidCell(input_size=64, hidden_size=128)
h_out = cell(torch.randn(2, 64))
assert h_out.shape == (2, 128)
print(f"      [OK] shape={h_out.shape}")

print("[2/5] Testing LiquidDiscriminator...")
from models.discriminator import LiquidDiscriminator
D = LiquidDiscriminator(img_channels=1, hidden_size=128)
score = D(torch.randn(2, 1, 64, 64))
assert score.shape == (2, 1)
print(f"      [OK] shape={score.shape}, values={score.detach().squeeze().tolist()}")

print("[3/5] Testing NeuralODEGenerator (euler, fast)...")
from models.generator import NeuralODEGenerator
G = NeuralODEGenerator(input_channels=3, output_channels=1,
                       latent_dim=64, ode_method='euler', rtol=1e-1, atol=1e-1)
rgb_in = torch.randn(2, 3, 64, 64)
out    = G(rgb_in)
B, C, H, W = out.shape
assert C == 1 and B == 2
print(f"      [OK] output shape={out.shape}, NFE={G.ode_func.nfe}")

print("[4/5] Testing PhysicsLoss...")
from losses.physics_loss import PhysicsLoss
physics = PhysicsLoss(alpha=0.001)
loss_val = physics(torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64))
assert loss_val.item() >= 0
print(f"      [OK] physics loss = {loss_val.item():.6f}")

print("[5/5] Testing full forward + backward pass...")
adv_loss = nn.BCELoss()
rec_loss = nn.L1Loss()
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4)
real_th  = torch.randn(2, 1, H, W)
real_lbl = torch.ones(2, 1)
opt_G.zero_grad()
fake_th   = G(rgb_in)
d_score   = D(fake_th)
l_adv     = adv_loss(d_score, real_lbl)
l_rec     = rec_loss(fake_th, real_th)
l_phy     = physics(fake_th, real_th)
total     = l_adv + 10 * l_rec + l_phy
total.backward()
opt_G.step()
print(f"      [OK] total_loss={total.item():.6f}  (grads OK)")

print("\n" + "="*50)
print("  ALL TESTS PASSED - Pipeline is ready!")
print("="*50)
print("\nNext step: python train.py --test")
