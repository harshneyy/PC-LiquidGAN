#!/bin/bash
# train_all_unet.sh
# Bulk training script for PC-LiquidGAN ODE-UNet on all remaining domains.
# Total estimated time: ~6 hours on exactly this GPU.

# Activate the virtual environment
source venv/bin/activate

echo "================================================================"
echo " Starting Bulk ODE-UNet Training for Remaining Domains"
echo "================================================================"

# 1. Chilli (2,227 pairs) -> ~45 mins
echo -e "\n\n[1/4] Training CHILLI domain..."
python3 train_unet.py --dataset chilli --epochs 100 --batch_size 16 --warmup 10 --lambda_spec 0.05 --init_noise 0.1

# 2. CBSR (2,700 pairs) -> ~1 hour
echo -e "\n\n[2/4] Training CBSR domain..."
python3 train_unet.py --dataset cbsr --epochs 100 --batch_size 16 --warmup 10 --lambda_spec 0.05 --init_noise 0.1

# 3. Medical (4,500 pairs) -> ~2 hours
echo -e "\n\n[3/4] Training MEDICAL domain..."
python3 train_unet.py --dataset medical --epochs 100 --batch_size 16 --warmup 10 --lambda_spec 0.05 --init_noise 0.1

# 4. KAIST (4,500 pairs) -> ~2 hours
echo -e "\n\n[4/4] Training KAIST domain..."
python3 train_unet.py --dataset kaist --epochs 100 --batch_size 16 --warmup 10 --lambda_spec 0.05 --init_noise 0.1

echo "================================================================"
echo " All domains successfully trained with ODE-UNet!"
echo " The new checkpoints are in ./checkpoints_unet/"
echo "================================================================"
