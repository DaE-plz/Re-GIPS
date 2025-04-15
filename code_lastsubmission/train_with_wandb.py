import os
import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import wandb  # 导入 wandb
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

wandb.login(key='cedea35395d2663bca804cf5d7e9ea172e08b2ca')

def save_as_tif(image, output_file):
    image_np = image.cpu().detach().numpy()
    tifffile.imsave(output_file, image_np)
    print(f'Image saved as {output_file}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

user_dir = '/home/woody/iwi5/iwi5191h'
project_dir = os.path.join(user_dir, 'dl_gips_forschung')
log_path = os.path.join(project_dir, 'training_log.csv')
checkpoint_file = 'model_checkpoint.pth'
checkpoint_path = os.path.join(project_dir, 'checkpoints', checkpoint_file)
save_path = os.path.join(project_dir, 'train_image')

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
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

total_patients = len(dataset)
test_size = int(total_patients * 0.2)
train_val_size = total_patients - test_size
torch.manual_seed(42)
train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])
val_size = int(train_val_size * 0.2)
train_size = train_val_size - val_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

input_channel = 1
output_channel = 1
image_size = [500, 128, 128]
proj_size = [300, 180]

num_proj_ap = 1
start_angle_ap = -np.pi / 2
end_angle_ap = np.pi / 2

num_proj_aplt = 2
start_angle_aplt = -np.pi / 2
end_angle_aplt = np.pi

geometry_encoder = geometry_encoder.Geometry_Encoder(input_channel, output_channel)
texture_encoder = texture_encoder.Texture_Encoder(input_channel, output_channel)
geometry_transform = Geometry_transformation.GeometryTransformation(output_channel, image_size, proj_size, num_proj_ap, start_angle_ap, end_angle_ap, num_proj_aplt, start_angle_aplt, end_angle_aplt)
image_generator = Image_Generator.ImageGenerator(output_channel, output_channel)
image_discriminator = Image_discriminator.ImageDiscriminator()

dlgips_geometry_encoder = geometry_encoder.cuda()
dlgips_texture_encoder = texture_encoder.cuda()
dlgips_geometry_transform = geometry_transform.cuda()
dlgips_image_generator = image_generator.cuda()
dlgips_image_discriminator = image_discriminator.cuda()

optimizer_geometry_encoder = torch.optim.Adam(dlgips_geometry_encoder.parameters(), lr=0.0001)
optimizer_texture_encoder = torch.optim.Adam(dlgips_texture_encoder.parameters(), lr=0.0001)
optimizer_geometry_transform = torch.optim.Adam(dlgips_geometry_transform.parameters(), lr=0.0001)
optimizer_image_generator = torch.optim.Adam(dlgips_image_generator.parameters(), lr=0.0001)
optimizer_image_discriminator = torch.optim.Adam(dlgips_image_discriminator.parameters(), lr=0.0001)

cyc_weight, rec_weight, adv_weight = 0.01, 0.1, 0.01
adv_scale1, adv_scale2, adv_scale3 = 0.5, 0.3, 0.2

# 初始化 wandb
wandb.init(project='dl_gips_project', entity='your_entity')
wandb.config = {
    "learning_rate": 0.0001,
    "epochs": 200,
    "batch_size": 10,
    "cyc_weight": 0.01,
    "rec_weight": 0.1,
    "adv_weight": 0.01,
    "adv_scale1": 0.5,
    "adv_scale2": 0.3,
    "adv_scale3": 0.2
}

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    dlgips_geometry_encoder.load_state_dict(checkpoint['geometry_encoder_state_dict'])
    dlgips_texture_encoder.load_state_dict(checkpoint['texture_encoder_state_dict'])
    dlgips_geometry_transform.load_state_dict(checkpoint['geometry_transform_state_dict'])
    dlgips_image_generator.load_state_dict(checkpoint['image_generator_state_dict'])
    dlgips_image_discriminator.load_state_dict(checkpoint['image_discriminator_state_dict'])
    optimizer_geometry_encoder.load_state_dict(checkpoint['optimizer_geometry_encoder'])
    optimizer_texture_encoder.load_state_dict(checkpoint['optimizer_texture_encoder'])
    optimizer_geometry_transform.load_state_dict(checkpoint['optimizer_geometry_transform'])
    optimizer_image_generator.load_state_dict(checkpoint['optimizer_image_generator'])
    optimizer_image_discriminator.load_state_dict(checkpoint['optimizer_image_discriminator'])
    start_epoch = checkpoint['epoch'] + 1
    avg_val_loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {start_epoch - 1}")
else:
    epoch = 1
    print("No checkpoint found, starting from scratch.")

    while epoch < 200:
        train_loss_total = 0
        num_batches = 0
        for i, (ap_truth, lt_truth) in enumerate(tqdm.tqdm(train_loader)):
            ap_truth, lt_truth = ap_truth.to(device), lt_truth.to(device)
            geomery_features = dlgips_geometry_encoder(ap_truth)
            texture_features = dlgips_texture_encoder(ap_truth)
            transformed_features = dlgips_geometry_transform(geomery_features)
            transformed_features_ap, transformed_features_lt = torch.chunk(transformed_features, 2, dim=1)
            generated_image_ap = dlgips_image_generator(transformed_features_ap, texture_features)
            generated_image_lt = dlgips_image_generator(transformed_features_lt, texture_features)
            generated_original_image = dlgips_image_generator(geomery_features, texture_features)

            real_outputs1, real_outputs2, real_outputs3 = image_discriminator(lt_truth)
            fake_outputs1, fake_outputs2, fake_outputs3 = image_discriminator(generated_image_lt.detach())

            generated_ap_feature = dlgips_geometry_encoder(generated_image_ap)
            generated_lt_feature = dlgips_texture_encoder(generated_image_lt)

            loss_cyc = consistency_loss.total_consistency_loss(generated_original_image, ap_truth, generated_lt_feature, transformed_features_lt, generated_ap_feature, transformed_features_ap)
            loss_rec = reconstruction_loss.reconstruction_loss(generated_image_lt, lt_truth, generated_image_ap, ap_truth)

            batch_size = lt_truth.size(0)
            real_labels1 = torch.ones(batch_size, 1, 18, 11, device=device)
            fake_labels1 = torch.zeros(batch_size, 1, 18, 11, device=device)
            real_labels2 = torch.ones(batch_size, 1, 9, 5, device=device)
            fake_labels2 = torch.zeros(batch_size, 1, 9, 5, device=device)
            real_labels3 = torch.ones(batch_size, 1, 4, 2, device=device)
            fake_labels3 = torch.zeros(batch_size, 1, 4, 2, device=device)

            loss_adv1 = adversarial_loss.adversarial_loss(real_outputs1, fake_outputs1, real_labels1, fake_labels1)
            loss_adv2 = adversarial_loss.adversarial_loss(real_outputs2, fake_outputs2, real_labels2, fake_labels2)
            loss_adv3 = adversarial_loss.adversarial_loss(real_outputs3, fake_outputs3, real_labels3, fake_labels3)
            loss_adv = adv_scale1 * loss_adv1 + adv_scale2 * loss_adv2 + adv_scale3 * loss_adv3

            optimizer_geometry_encoder.zero_grad()
            optimizer_texture_encoder.zero_grad()
            optimizer_geometry_transform.zero_grad()
            optimizer_image_generator.zero_grad()
            optimizer_image_discriminator.zero_grad()

            total_loss = cyc_weight * loss_cyc + rec_weight * loss_rec + adv_weight * loss_adv
            total_loss.backward()
            optimizer_geometry_encoder.step()
            optimizer_texture_encoder.step()
            optimizer_geometry_transform.step()
            optimizer_image_generator.step()
            optimizer_image_discriminator.step()

            if i % 10 == 0:
                _generated_image_ap = generated_image_ap[0].cpu().detach()
                _generated_image_lt = generated_image_lt[0].cpu().detach()
                _generated_image_original = generated_original_image[0].cpu().detach()
                save_as_tif(_generated_image_ap, f'{save_path}/{epoch}_{i}_generated_ap.tif')
                save_as_tif(_generated_image_lt, f'{save_path}/{epoch}_{i}_generated_lt.tif')
                save_as_tif(_generated_image_original, f'{save_path}/{epoch}_{i}_generated_original.tif')

            train_loss_total += total_loss.item()
            num_batches += 1

        avg_train_loss = train_loss_total / num_batches
        wandb.log({"avg_train_loss": avg_train_loss})  # 记录训练损失

        dlgips_geometry_encoder.eval()
        dlgips_texture_encoder.eval()
        dlgips_geometry_transform.eval()
        dlgips_image_generator.eval()
        dlgips_image_discriminator.eval()

        with torch.no_grad():
            val_loss_total = 0
            for ap_truth, lt_truth in val_loader:
                ap_truth, lt_truth = ap_truth.to(device), lt_truth.to(device)
                geomery_features = dlgips_geometry_encoder(ap_truth)
                texture_features = dlgips_texture_encoder(ap_truth)
                transformed_features = dlgips_geometry_transform(geomery_features)
                transformed_features_ap, transformed_features_lt = torch.chunk(transformed_features, 2, dim=1)
                generated_image_ap = dlgips_image_generator(transformed_features_ap, texture_features)
                generated_image_lt = dlgips_image_generator(transformed_features_lt, texture_features)
                generated_original_image = dlgips_image_generator(geomery_features, texture_features)

                real_outputs1, real_outputs2, real_outputs3 = image_discriminator(lt_truth)
                fake_outputs1, fake_outputs2, fake_outputs3 = image_discriminator(generated_image_lt.detach())

                loss_cyc = consistency_loss.total_consistency_loss(generated_original_image, ap_truth, generated_lt_feature, transformed_features_lt, generated_ap_feature, transformed_features_ap)
                loss_rec = reconstruction_loss.reconstruction_loss(generated_image_lt, lt_truth, generated_image_ap, ap_truth)

                loss_adv1 = adversarial_loss.adversarial_loss(real_outputs1, fake_outputs1, real_labels1, fake_labels1)
                loss_adv2 = adversarial_loss.adversarial_loss(real_outputs2, fake_outputs2, real_labels2, fake_labels2)
                loss_adv3 = adversarial_loss.adversarial_loss(real_outputs3, fake_outputs3, real_labels3, fake_labels3)
                loss_adv = adv_scale1 * loss_adv1 + adv_scale2 * loss_adv2 + adv_scale3 * loss_adv3

                val_loss = cyc_weight * loss_cyc + rec_weight * loss_rec + adv_weight * loss_adv
                val_loss_total += val_loss.item()

            avg_val_loss = val_loss_total / len(val_loader)
            wandb.log({"avg_val_loss": avg_val_loss})  # 记录验证损失

        with open(log_path, 'a') as log_file:
            log_file.write(f'{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f}\n')

        print(f'Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')

        if epoch % 20 == 0:
            checkpoint = {
                'epoch': epoch,
                'geometry_encoder_state_dict': dlgips_geometry_encoder.state_dict(),
                'texture_encoder_state_dict': dlgips_texture_encoder.state_dict(),
                'geometry_transform_state_dict': dlgips_geometry_transform.state_dict(),
                'image_generator_state_dict': dlgips_image_generator.state_dict(),
                'image_discriminator_state_dict': dlgips_image_discriminator.state_dict(),
                'optimizer_geometry_encoder': optimizer_geometry_encoder.state_dict(),
                'optimizer_texture_encoder': optimizer_texture_encoder.state_dict(),
                'optimizer_geometry_transform': optimizer_geometry_transform.state_dict(),
                'optimizer_image_generator': optimizer_image_generator.state_dict(),
                'optimizer_image_discriminator': optimizer_image_discriminator.state_dict(),
                'loss': avg_val_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch}')

        epoch += 1