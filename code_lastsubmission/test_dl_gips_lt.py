import os
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import projection_dataset
import tifffile
import numpy as np
import geometry_encoder
import texture_encoder
import Image_Generator
import Geometry_transformation
import consistency_loss
import reconstruction_loss
import Image_discriminator
import adversarial_loss
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision import transforms

# for test
from torch.nn.functional import mse_loss
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# test
def apply_window(image, window_level, window_width):
    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)

    windowed_image = torch.clamp(image, min=lower_bound, max=upper_bound)
    windowed_image = (windowed_image - lower_bound) / window_width * 255  # Scale to (0, 255) range if needed

    return windowed_image


def calculate_metrics(pred, truth):
    # Convert torch tensors to numpy arrays and squeeze batch and channel dimensions
    pred_np = pred.squeeze().cpu().detach().numpy()  # Squeeze to [H, W]
    truth_np = truth.squeeze().cpu().detach().numpy()  # Squeeze to [H, W]

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(pred_np - truth_np))

    # Calculate Root Mean Squared Error (RMSE) using PyTorch, then convert to numpy
    rmse = np.sqrt(torch.nn.functional.mse_loss(pred, truth, reduction='mean').item())

    # Determine the smallest dimension of your images
    smallest_side = min(truth_np.shape)
    win_size = max(3, smallest_side // 2 * 2 + 1)  # Ensure it is odd and less than the smallest image dimension

    # Calculate SSIM
    ssim_index = ssim(truth_np, pred_np, win_size=win_size, data_range=truth_np.max() - truth_np.min())

    # Calculate PSNR
    psnr_value = psnr(truth_np, pred_np, data_range=truth_np.max() - truth_np.min())

    return mae, rmse, ssim_index, psnr_value


def save_as_tif(image, output_file):
    # Make sure image is a numpy array
    image_np = image.cpu().detach().numpy()
    # Make sure the image is in the correct format (H x W or H x W x C)
    if image_np.ndim == 4:
        # If the image tensor has shape (1, C, H, W), convert it to (H, W, C)
        image_np = np.squeeze(image_np, axis=0).transpose(1, 2, 0)
    elif image_np.ndim == 3 and image_np.shape[0] == 1:
        # If the image tensor has shape (1, H, W), get rid of the first dimension
        image_np = np.squeeze(image_np, axis=0)
    # Save the image
    tifffile.imwrite(output_file, image_np)  # Ensure parameter order is correct



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


user_dir = '/home/woody/iwi5/iwi5191h'
project_dir = os.path.join(user_dir, 'dl_gips_forschung')

log_path = os.path.join(project_dir, 'training_log.csv')
checkpoint_file = 'model_checkpoint.pth'
checkpoint_path = os.path.join(project_dir, 'checkpoints', checkpoint_file)  # 指向具体文件
save_path = os.path.join(project_dir, 'test_image_lt')


test_metrics_path = os.path.join(project_dir, 'test_metrics.csv')


if not os.path.exists(test_metrics_path):
    with open(test_metrics_path, 'w') as metric_file:
        metric_file.write('Index,MAE,RMSE,SSIM,PSNR\n')
#log_path = 'training_log.csv'
if not os.path.exists(log_path):
    with open(log_path, 'w') as log_file:
        log_file.write('Epoch,Train Loss,Validation Loss\n')

#checkpoint_path = 'checkpoints/dl_gips_checkpoint.pth'
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

#save_path = 'train_image'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 传入数据集，包含ap和lt角度的ground_truth 投影
#dataset = hpc_code.dataset_prepare.projection_dataset.ProjectionTensorDataset(ap_dir='G:/dataset/projection_ap', lt_dir='G:/dataset/projection_lt')
# dataset =projection_dataset.ProjectionTensorDataset(
#     ap_dir='/home/hpc/iwi5/iwi5191h/dataset/projection_ap',
#     lt_dir='/home/hpc/iwi5/iwi5191h/dataset/projection_lt'
# )


dataset = projection_dataset.ProjectionTensorDataset(
    ap_dir=f'/scratch/{os.environ["SLURM_JOB_ID"]}/ap',
    lt_dir=f'/scratch/{os.environ["SLURM_JOB_ID"]}/lt',

)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 计算训练/验证集和测试集的大小
total_patients = len(dataset)
test_size = int(total_patients * 0.2)
train_val_size = total_patients - test_size

# 定义随机种子以确保可复现性 # 保证每次运行代码时，所有的随机操作（比如初始化权重、打乱数据集、随机选择dropout单元等）都会以相同的方式发生
torch.manual_seed(42)

# 随机划分数据集为训练/验证集和测试集
train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])


# 将20%的训练/验证集数据用作验证
val_size = int(train_val_size * 0.2)
train_size = train_val_size - val_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# 创建相应的 DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# Print the number of patients in the test loader
print(f"Number of patients in the test loader: {len(test_loader)}")




input_channel = 1
output_channel =1
image_size = [500, 128, 128]
proj_size = [300, 180]

num_proj_ap = 1
start_angle_ap = -np.pi / 2
end_angle_ap = np.pi / 2

num_proj_aplt = 2
start_angle_aplt = -np.pi / 2
end_angle_aplt = np.pi

geometry_encoder = geometry_encoder.Geometry_Encoder(input_channel,output_channel)
texture_encoder = texture_encoder.Texture_Encoder(input_channel,output_channel)
geometry_transform = Geometry_transformation.GeometryTransformation(output_channel, image_size,
                                                                              proj_size,
                                                                              num_proj_ap,start_angle_ap,end_angle_ap,num_proj_aplt,start_angle_aplt,end_angle_aplt)
image_generator = Image_Generator.ImageGenerator(output_channel,output_channel)
image_discriminator=Image_discriminator.ImageDiscriminator()

dlgips_geometry_encoder = geometry_encoder.cuda()    # ε_g
dlgips_texture_encoder = texture_encoder.cuda()      # ε_t
dlgips_geometry_transform = geometry_transform.cuda()  # BP&Net&FP
dlgips_image_generator = image_generator.cuda()     # G
dlgips_image_discriminator = image_discriminator.cuda()  # D

# 定义优化器和超参数
optimizer_geometry_encoder = torch.optim.Adam(dlgips_geometry_encoder.parameters(), lr=0.0001,betas=(0.5, 0.999), weight_decay=1e-5)
optimizer_texture_encoder = torch.optim.Adam(dlgips_texture_encoder.parameters(), lr=0.0001,betas=(0.5, 0.999), weight_decay=1e-5)

optimizer_geometry_transform = torch.optim.Adam(dlgips_geometry_transform.parameters(), lr=0.0001,betas=(0.5, 0.999), weight_decay=1e-5)
optimizer_image_generator = torch.optim.Adam(dlgips_image_generator.parameters(), lr=0.0001,betas=(0.5, 0.999), weight_decay=1e-5)
optimizer_image_discriminator = torch.optim.Adam(dlgips_image_discriminator.parameters(), lr=0.0001,betas=(0.5, 0.999), weight_decay=1e-5)

cyc_weight, rec_weight, adv_weight = 0.01, 0.1, 0.01
adv_scale1,adv_scale2,adv_scale3=0.5,0.3,0.2

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    # 加载所有模块的状态字典
    dlgips_geometry_encoder.load_state_dict(checkpoint['geometry_encoder_state_dict'])
    dlgips_texture_encoder.load_state_dict(checkpoint['texture_encoder_state_dict'])
    dlgips_geometry_transform.load_state_dict(checkpoint['geometry_transform_state_dict'])
    dlgips_image_generator.load_state_dict(checkpoint['image_generator_state_dict'])
    dlgips_image_discriminator.load_state_dict(checkpoint['image_discriminator_state_dict'])
    # 加载所有优化器的状态
    optimizer_geometry_encoder.load_state_dict(checkpoint['optimizer_geometry_encoder'])
    optimizer_texture_encoder.load_state_dict(checkpoint['optimizer_texture_encoder'])
    optimizer_geometry_transform.load_state_dict(checkpoint['optimizer_geometry_transform'])
    optimizer_image_generator.load_state_dict(checkpoint['optimizer_image_generator'])
    optimizer_image_discriminator.load_state_dict(checkpoint['optimizer_image_discriminator'])
    print('Model loaded successfully')
else:
    print('No checkpoint found, starting from scratch.')
    exit()

dlgips_geometry_encoder.eval()
dlgips_texture_encoder.eval()
dlgips_geometry_transform.eval()
dlgips_image_generator.eval()
dlgips_image_discriminator.eval()

with open(test_metrics_path, 'w') as metric_file:
    metric_file.write('Index,MAE,RMSE,SSIM,PSNR\n')
with torch.no_grad():
    for idx, (ap_truth, lt_truth) in tqdm(enumerate(test_loader), total=len(test_loader)):
        ap_truth, lt_truth = ap_truth.to(device), lt_truth.to(device)

        # Use lt_truth instead of ap_truth for feature extraction
        geometry_features = dlgips_geometry_encoder(lt_truth)  # f_src_g now derived from lt_truth
        texture_features = dlgips_texture_encoder(lt_truth)  # f_t now derived from lt_truth

        # Process the features as before
        transformed_features = dlgips_geometry_transform(geometry_features)
        transformed_features_ap, transformed_features_lt = torch.chunk(transformed_features, 2, dim=1)

        # Generate projections from the transformed features
        generated_image_ap = dlgips_image_generator(transformed_features_ap, texture_features)
        generated_image_lt = dlgips_image_generator(transformed_features_lt, texture_features)

        # Discriminator and feature extraction on the generated images (no change needed here)
        real_outputs1, real_outputs2, real_outputs3 = image_discriminator(lt_truth)
        fake_outputs1, fake_outputs2, fake_outputs3 = image_discriminator(generated_image_lt.detach())

        generated_ap_feature = dlgips_geometry_encoder(generated_image_ap)
        generated_lt_feature = dlgips_texture_encoder(generated_image_lt)
        window_level = 2.5  # Window level for (0, 5) window
        window_width = 5.0  # Window width for (0, 5) window
        # Apply intensity window and save images (no change needed here)
        lt_truth_windowed = apply_window(lt_truth, window_level, window_width)
        generated_image_ap_windowed = apply_window(generated_image_ap, window_level, window_width)
        generated_image_lt_windowed = apply_window(generated_image_lt, window_level, window_width)
        save_as_tif(lt_truth_windowed, os.path.join(save_path, f'lt_truth_{idx}_windowed.tif'))
        save_as_tif(generated_image_ap_windowed, os.path.join(save_path, f'generated_image_ap_{idx}_windowed.tif'))
        save_as_tif(generated_image_lt_windowed, os.path.join(save_path, f'generated_image_lt_{idx}_windowed.tif'))

        # Calculate and log metrics
        mae, rmse, ssim_index, psnr_value = calculate_metrics(generated_image_lt, lt_truth)
        with open(test_metrics_path, 'a') as metric_file:
            metric_file.write(f'{idx},{mae},{rmse},{ssim_index},{psnr_value}\n')