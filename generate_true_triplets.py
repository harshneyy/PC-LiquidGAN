import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from models.generator import ODEUNetGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets = ['medical', 'cbsr', 'agri', 'chilli', 'kaist']

paths = {
    'medical': ('datasets/medical/train/rgb/image_0001.png', 'datasets/medical/train/thermal/image_0001.png'),
    'cbsr': ('datasets/cbsr/train/rgb/0001_01.jpg', 'datasets/cbsr/train/thermal/0001_01.jpg'),
    'agri': ('datasets/agri/train/rgb/tomato_001.png', 'datasets/agri/train/thermal/tomato_001.png'),
    'chilli': ('datasets/chilli/train/rgb/chilli_001.png', 'datasets/chilli/train/thermal/chilli_001.png'),
    'kaist': ('datasets/kaist/train/rgb/I00000.jpg', 'datasets/kaist/train/thermal/I00000.jpg')
}

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

rows = []

for ds in datasets:
    print(f"Generating triplet for {ds}...", flush=True)
    # 1. Load Generator
    G = ODEUNetGenerator(input_channels=3, output_channels=1).to(device)
    ckpt = torch.load(f'checkpoints_unet/{ds}/ckpt_epoch_0100.pth', map_location=device, weights_only=True)
    G.load_state_dict(ckpt['G_state'])
    G.eval()
    
    # 2. Find any valid image in the dataset directory
    rgb_dir = f'data/{ds}/val/rgb'
    therm_dir = f'data/{ds}/val/thermal'
    
    rgb_files = sorted(os.listdir(rgb_dir))
    therm_files = sorted(os.listdir(therm_dir))
    
    if not rgb_files:
        print(f"No files found for {ds}")
        continue
        
    rgb_path = os.path.join(rgb_dir, rgb_files[0])
    therm_path = os.path.join(therm_dir, therm_files[0])
    
    img_rgb = Image.open(rgb_path).convert('RGB')
    img_therm = Image.open(therm_path).convert('L')
    
    # Convert rgb to tensor
    tensor_rgb = transform(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        tensor_fake = G(tensor_rgb)
        
    # Denormalize
    tensor_rgb = tensor_rgb.squeeze().cpu() * 0.5 + 0.5
    tensor_fake = tensor_fake.squeeze().cpu() * 0.5 + 0.5
    
    # Convert back to PIL
    to_pil = transforms.ToPILImage()
    
    img_rgb_resized = to_pil(tensor_rgb).resize((256, 256))
    img_therm_resized = img_therm.resize((256, 256))
    img_fake_resized = to_pil(tensor_fake).resize((256, 256))
    
    # Ensure all are explicitly RGB for pasting into the grid
    img_rgb_resized = img_rgb_resized.convert('RGB')
    img_therm_resized = img_therm_resized.convert('RGB')
    img_fake_resized = img_fake_resized.convert('RGB')
    
    row_img = Image.new('RGB', (256 * 3, 256))
    row_img.paste(img_rgb_resized, (0, 0))
    row_img.paste(img_therm_resized, (256, 0))
    row_img.paste(img_fake_resized, (512, 0))
    
    rows.append(row_img)

if rows:
    grid_img = Image.new('RGB', (256 * 3, 256 * len(rows)))
    y_offset = 0
    for r in rows:
        grid_img.paste(r, (0, y_offset))
        y_offset += 256
        
    grid_img.save('qualitative_grid.png')
    print('Generated perfect true triplets into qualitative_grid.png', flush=True)
