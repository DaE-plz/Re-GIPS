import os
import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import projection_dataset
import u_net
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import tifffile
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_model(model, data_loader, device, save_path, epoch):
    model.eval()
    mse_loss = nn.MSELoss(reduction='sum')
    mae_loss = nn.L1Loss(reduction='sum')
    total_mse = 0.0
    total_mae = 0.0
    total_ssim = 0.0
    total_psnr = 0.0  # Initialize total PSNR
    total_samples = 0

    with torch.no_grad():
        for idx, (ap_projection, lt_projection) in enumerate(data_loader):
            ap_projection, lt_projection = ap_projection.to(device), lt_projection.to(device)
            lt_pred = model(ap_projection)

            mse = mse_loss(lt_pred, lt_projection)
            mae = mae_loss(lt_pred, lt_projection)
            total_mse += mse.item()
            total_mae += mae.item()

            # Convert tensors to numpy arrays for metric calculations
            lt_projection_np = lt_projection.squeeze().cpu().numpy()
            lt_pred_np = lt_pred.squeeze().cpu().numpy()

            # Calculate SSIM and PSNR for each image in the batch
            for i in range(lt_pred_np.shape[0]):
                total_ssim += ssim(lt_pred_np[i], lt_projection_np[i], data_range=lt_projection_np[i].max() - lt_projection_np[i].min())
                total_psnr += psnr(lt_projection_np[i], lt_pred_np[i], data_range=lt_projection_np[i].max() - lt_projection_np[i].min())

            total_samples += lt_projection.size(0)

            # Save images every 10 batches or as needed
            if idx % 10 == 0:  # Adjust based on how frequently you want to save the images
                tifffile.imwrite(os.path.join(save_path, f'{epoch}_ap_truth_{idx}.tif'), ap_projection.squeeze().cpu().numpy())
                tifffile.imwrite(os.path.join(save_path, f'{epoch}_lt_truth_{idx}.tif'), lt_projection_np[0])
                tifffile.imwrite(os.path.join(save_path, f'{epoch}_lt_pred_{idx}.tif'), lt_pred_np[0])

    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    avg_ssim = total_ssim / total_samples
    avg_psnr = total_psnr / total_samples  # Calculate average PSNR
    return avg_mse, avg_mae, avg_ssim, avg_psnr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

user_dir = '/home/woody/iwi5/iwi5191h'
project_dir = os.path.join(user_dir, 'dl_gips_forschung')

log_path = os.path.join(project_dir, 'training_log.csv')
checkpoint_file = 'model_checkpoint.pth'
checkpoint_path = os.path.join(project_dir, 'checkpoints', checkpoint_file)
save_path = os.path.join(project_dir, 'test_image')

if not os.path.exists(log_path):
    with open(log_path, 'w') as log_file:
        log_file.write('Epoch,Train Loss,Validation Loss\n')

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

if not os.path.exists(save_path):
    os.makedirs(save_path)

dataset = projection_dataset.ProjectionTensorDataset(
    ap_dir=f'/scratch/{os.environ["SLURM_JOB_ID"]}/ap',
    lt_dir=f'/scratch/{os.environ["SLURM_JOB_ID"]}/lt'
)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

total_patients = len(dataset)
test_size = int(total_patients * 0.2)
train_val_size = total_patients - test_size

torch.manual_seed(42)
train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])

val_size = int(train_val_size * 0.2)
train_size = train_val_size - val_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

net = u_net.UNet(num_classes=1).to(device)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if 'net_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['net_state_dict'])
        print('Model state loaded successfully.')
    else:
        print('No model state found in checkpoint.')
else:
    print('No checkpoint found, starting from scratch.')

# Evaluate the model and save test images
epoch = 1  # Assuming evaluation for a single epoch; adjust as needed
test_mse, test_mae, test_ssim, test_psnr = evaluate_model(net, test_loader, device, save_path, epoch)
print(f'Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}, Test SSIM: {test_ssim:.4f}')

# Save test results to a file
results_path = os.path.join(project_dir, 'test_results.txt')
with open(results_path, 'w') as f:
    f.write(f'Test MSE: {test_mse:.4f}\n')
    f.write(f'Test MAE: {test_mae:.4f}\n')
    f.write(f'Test SSIM: {test_ssim:.4f}\n')
    f.write(f'Test PSNR: {test_psnr:.4f}\n')
print(f'Test results saved to {results_path}')