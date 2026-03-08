import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os

def create_bar_chart():
    models = ['WGAN-GP', 'DCGAN', 'Base LiquidGAN', 'PC-LiquidGAN\n(Ours)']
    ssim = [0.1398, 0.5945, 0.8222, 0.9945]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, ssim, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('SSIM Score', fontsize=14, fontweight='bold')
    ax.set_title('Ablation Study: Model Performance (Agri Dataset)', fontsize=16, fontweight='bold')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results_graph.png', dpi=300)
    plt.close()
    print("Created results_graph.png")

def create_qualitative_grid():
    datasets = ['medical', 'cbsr', 'agri']
    rows = []
    
    for ds in datasets:
        img_path = f'results_unet/{ds}/epoch_0100.png'
        if os.path.exists(img_path):
            img = Image.open(img_path)
            w, h = img.size
            row_h = h // 2
            col_w = w // 4
            
            # Crop Input, GT, Fake
            inp = img.crop((0, 0, col_w, row_h))
            gt = img.crop((col_w, 0, col_w*2, row_h))
            fake = img.crop((col_w*2, 0, col_w*3, row_h))
            
            # Concatenate horizontally
            row_img = Image.new('RGB', (col_w * 3, row_h))
            row_img.paste(inp, (0, 0))
            row_img.paste(gt, (col_w, 0))
            row_img.paste(fake, (col_w*2, 0))
            rows.append(row_img)
    
    if rows:
        grid_w = rows[0].width
        grid_h = sum(r.height for r in rows)
        grid_img = Image.new('RGB', (grid_w, grid_h))
        
        y_offset = 0
        for r in rows:
            grid_img.paste(r, (0, y_offset))
            y_offset += r.height
            
        grid_img.save('qualitative_grid.png')
        print("Created qualitative_grid.png")

def create_architecture():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Draw blocks
    ax.add_patch(patches.Rectangle((0.0, 0.4), 0.15, 0.2, fill=True, color='lightblue', ec='black'))
    ax.text(0.075, 0.5, 'Input\n(RGB / NIR)', ha='center', va='center', fontweight='bold')
    
    ax.add_patch(patches.Rectangle((0.25, 0.3), 0.15, 0.4, fill=True, color='lightgreen', ec='black'))
    ax.text(0.325, 0.5, 'UNet\nEncoder', ha='center', va='center', fontweight='bold')
    
    ax.add_patch(patches.Rectangle((0.5, 0.35), 0.2, 0.3, fill=True, color='orange', ec='black'))
    ax.text(0.6, 0.5, 'ConvODEFunc\n(dopri5 Adaptive)', ha='center', va='center', fontweight='bold')
    
    ax.add_patch(patches.Rectangle((0.8, 0.3), 0.15, 0.4, fill=True, color='lightgreen', ec='black'))
    ax.text(0.875, 0.5, 'UNet\nDecoder', ha='center', va='center', fontweight='bold')
    
    ax.add_patch(patches.Rectangle((1.05, 0.4), 0.15, 0.2, fill=True, color='pink', ec='black'))
    ax.text(1.125, 0.5, 'Output\n(Thermal)', ha='center', va='center', fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(0.25,0.5), xytext=(0.15,0.5), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
    ax.annotate('', xy=(0.5,0.5), xytext=(0.4,0.5), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
    ax.annotate('', xy=(0.8,0.5), xytext=(0.7,0.5), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
    ax.annotate('', xy=(1.05,0.5), xytext=(0.95,0.5), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
    
    # Skip connection
    ax.annotate('', xy=(0.875, 0.7), xytext=(0.325, 0.7), arrowprops=dict(facecolor='black', shrink=0.0, width=1, headwidth=6, connectionstyle="arc3,rad=-0.4"))
    ax.text(0.6, 0.88, 'Spatial Skip Connection\n(Preserves Details)', ha='center', va='center', fontweight='bold', color='red')
    
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1.2)
    plt.tight_layout()
    plt.savefig('architecture.png', dpi=300)
    plt.close()
    print("Created architecture.png")

if __name__ == '__main__':
    create_bar_chart()
    create_qualitative_grid()
    create_architecture()
