import os
import csv
from tensorboard.backend.event_processing import event_accumulator

LOG_DIR = './logs'

def extract_metrics(log_file):
    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()
    
    metrics = {'file': os.path.basename(log_file)}
    
    try:
        # Check available tags
        tags = ea.Tags()['scalars']
        
        if 'Metrics/SSIM' in tags:
            metrics['SSIM'] = ea.Scalars('Metrics/SSIM')[-1].value
        if 'Metrics/PSNR' in tags:
            metrics['PSNR'] = ea.Scalars('Metrics/PSNR')[-1].value
            
        if 'Epoch/G_loss' in tags:
            g_losses = ea.Scalars('Epoch/G_loss')
            metrics['G_loss'] = g_losses[-1].value
            metrics['Epochs'] = len(g_losses)
            
    except Exception as e:
        print(f"Error processing {log_file}: {e}")
        
    return metrics

def run():
    files = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if 'tfevents' in f]
    files.sort(key=os.path.getmtime)
    
    all_results = []
    for f in files:
        m = extract_metrics(f)
        all_results.append(m)
        print(f"Extracted from {m['file']}: SSIM={m.get('SSIM', 'N/A')}, PSNR={m.get('PSNR', 'N/A')}")

    if not all_results:
        print("No metrics found.")
        return

    # Write to CSV using standard csv module
    fieldnames = ['file', 'Epochs', 'SSIM', 'PSNR', 'G_loss']
    with open('results/extracted_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    
    print("\nResults saved to results/extracted_metrics.csv")

if __name__ == '__main__':
    run()
