import os
import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import code.dataset_prepare.projection_dataset
import tifffile
import numpy as np
import code.module_dl_gips.geometry_encoder
import code.module_dl_gips.texture_encoder
import code.module_dl_gips.Image_Generator
import code.module_dl_gips.Geometry_transformation
import consistency_loss
import reconstruction_loss
import matplotlib.pyplot as plt
from torch.utils.data import random_split


def save_as_tif(image, output_file):
    # 确保image是numpy数组
    image_np = image.cpu().detach().numpy()
    tifffile.imsave(output_file, image_np)  # 确保参数顺序正确
    print(f'Image saved as {output_file}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_path = 'training_log.csv'
if not os.path.exists(log_path):
    with open(log_path, 'w') as log_file:
        log_file.write('Epoch,Train Loss,Validation Loss\n')

checkpoint_path = 'checkpoints/dl_gips_checkpoint.pth'
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

save_path = 'train_image'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 传入数据集，包含ap和lt角度的ground_truth 投影
dataset = code.dataset_prepare.projection_dataset.ProjectionTensorDataset(ap_dir='G:/dataset/projection_ap', lt_dir='G:/dataset/projection_lt')
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 计算训练/验证集和测试集的大小
total_patients = len(dataset)  # 使用您数据集的实际大小
test_size = int(total_patients * 0.2)
train_val_size = total_patients - test_size

# 定义随机种子以确保可复现性 # 保证每次运行代码时，所有的随机操作（比如初始化权重、打乱数据集、随机选择dropout单元等）都会以相同的方式发生
torch.manual_seed(42)

# 随机划分数据集为训练/验证集和测试集
train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])


# 将20%的训练/验证集数据用作验证，则：
val_size = int(train_val_size * 0.2)
train_size = train_val_size - val_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# 创建相应的 DataLoader
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)



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

geometry_encoder = code.module_dl_gips.geometry_encoder.Geometry_Encoder(input_channel,output_channel)
texture_encoder = code.module_dl_gips.texture_encoder.Texture_Encoder(input_channel,output_channel)
geometry_transform = code.module_dl_gips.Geometry_transformation.GeometryTransformation(output_channel, image_size,
                                                                              proj_size,
                                                                              num_proj_ap,start_angle_ap,end_angle_ap,num_proj_aplt,start_angle_aplt,end_angle_aplt)
image_generator = code.module_dl_gips.Image_Generator.ImageGenerator(output_channel,output_channel)

dlgips_geometry_encoder = geometry_encoder.cuda()    # ε_g
dlgips_texture_encoder = texture_encoder.cuda()      # ε_t
dlgips_geometry_transform = geometry_transform.cuda()  # BP&Net&FP
dlgips_image_generator = image_generator.cuda()     # G
#dlgips_image_discriminator = image_discriminator.cuda()  # D

# 定义优化器和超参数
optimizer_geometry_encoder = torch.optim.Adam(dlgips_geometry_encoder.parameters(), lr=0.0001)
optimizer_texture_encoder = torch.optim.Adam(dlgips_texture_encoder.parameters(), lr=0.0001)

optimizer_geometry_transform = torch.optim.Adam(dlgips_geometry_transform.parameters(), lr=0.0001)
optimizer_image_generator = torch.optim.Adam(dlgips_image_generator.parameters(), lr=0.0001)

cyc_weight, rec_weight, adv_weight = 0.01, 0.1, 0.01


if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    # 加载所有模块的状态字典
    dlgips_geometry_encoder.load_state_dict(checkpoint['geometry_encoder_state_dict'])
    dlgips_texture_encoder.load_state_dict(checkpoint['texture_encoder_state_dict'])
    dlgips_geometry_transform.load_state_dict(checkpoint['geometry_transform_state_dict'])
    dlgips_image_generator.load_state_dict(checkpoint['image_generator_state_dict'])

    # 加载所有优化器的状态
    optimizer_geometry_encoder.load_state_dict(checkpoint['optimizer_geometry_encoder'])
    optimizer_texture_encoder.load_state_dict(checkpoint['optimizer_texture_encoder'])
    optimizer_geometry_transform.load_state_dict(checkpoint['optimizer_geometry_transform'])
    optimizer_image_generator.load_state_dict(checkpoint['optimizer_image_generator'])

    # 加载其他信息，如当前epoch
    start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始继续训练
    avg_val_loss = checkpoint['loss']

    print(f"Loaded checkpoint from epoch {start_epoch - 1}")
else:
    epoch = 1
    print("No checkpoint found, starting from scratch.")

    while epoch < 200:
        train_loss_total = 0  # 累计整个epoch的损失
        num_batches = 0  # 记录批次数量
        for i, (ap_truth, lt_truth) in enumerate(tqdm.tqdm(train_loader)):
            ap_truth, lt_truth =ap_truth.to(device), lt_truth.to(device)

            geomery_features = dlgips_geometry_encoder(ap_truth)  # f_src_g
            texture_features = dlgips_texture_encoder(ap_truth)  # f_t

            transformed_features = dlgips_geometry_transform(geomery_features)
            # F_src_g                   F_tgt_g
            transformed_features_ap, transformed_features_lt = torch.chunk(transformed_features, 2, dim=1)

            # generator 生成 不同角度的投影
            generated_image_ap = dlgips_image_generator(transformed_features_ap, texture_features)  # I_src'
            generated_image_lt = dlgips_image_generator(transformed_features_lt, texture_features)  # I_tgt'
            generated_original_image = dlgips_image_generator(geomery_features, texture_features)  # G(f_src_g,f_t)

            # 将 生成的投影  再放入 feature encoder中，得到 features
            generated_ap_feature = dlgips_geometry_encoder(generated_image_ap)  # ε_g(I_src')
            generated_lt_feature = dlgips_texture_encoder(generated_image_lt)  # ε_g(I_tgt')

            loss_cyc = consistency_loss.total_consistency_loss(generated_original_image,
                                                               ap_truth, generated_lt_feature,
                                                               transformed_features_lt,
                                                               generated_ap_feature, transformed_features_ap)
            loss_rec = reconstruction_loss.reconstruction_loss(generated_image_lt, lt_truth,
                                                               generated_image_ap,
                                                               ap_truth)

            optimizer_geometry_encoder.zero_grad()
            optimizer_texture_encoder.zero_grad()
            optimizer_geometry_transform.zero_grad()
            optimizer_image_generator.zero_grad()

            total_loss = cyc_weight * loss_cyc + rec_weight * loss_rec



            total_loss.backward()
            optimizer_geometry_encoder.step()
            optimizer_texture_encoder.step()
            optimizer_geometry_transform.step()
            optimizer_image_generator.step()

            # if i % 1 == 0:
            #     print(f'{epoch}-{i}-total_loss===>>{total_loss.item()}')

            if i % 10 == 0:  # 每10个batch保存一次图像
                # 将生成的图像从Tensor转换为numpy数组，并缩放到[0, 255]范围
                _generated_image_ap = generated_image_ap[0].cpu().detach() * 255
                _generated_image_lt = generated_image_lt[0].cpu().detach() * 255
                _generated_image_original = generated_original_image[0].cpu().detach() * 255


                # 使用tifffile保存图像
                save_as_tif(_generated_image_ap, f'{save_path}/{epoch}_{i}_generated_ap.tif')
                save_as_tif(_generated_image_lt, f'{save_path}/{epoch}_{i}_generated_lt.tif')
                save_as_tif(_generated_image_original, f'{save_path}/{epoch}_{i}_generated_original.tif')

            # 累计损失和批次数量
            train_loss_total += total_loss.item()
            num_batches += 1

        # 计算并打印平均训练损失
        avg_train_loss = train_loss_total / num_batches
        # print(f'Epoch {epoch}, Average Training Loss: {avg_train_loss}')


        # Validation phase after each training epoch
        dlgips_geometry_encoder.eval()
        dlgips_texture_encoder.eval()
        dlgips_geometry_transform.eval()
        dlgips_image_generator.eval()


        with torch.no_grad():
            val_loss_total = 0
            for ap_truth, lt_truth in val_loader:
                ap_truth, lt_truth = ap_truth.to(device), lt_truth.to(device)

                geomery_features = dlgips_geometry_encoder(ap_truth)  # f_src_g
                texture_features = dlgips_texture_encoder(ap_truth)  # f_t

                transformed_features = dlgips_geometry_transform(geomery_features)
                # F_src_g                   F_tgt_g
                transformed_features_ap, transformed_features_lt = torch.chunk(transformed_features, 2, dim=1)

                # generator 生成 不同角度的投影
                generated_image_ap = dlgips_image_generator(transformed_features_ap, texture_features)  # I_src'
                generated_image_lt = dlgips_image_generator(transformed_features_lt, texture_features)  # I_tgt'
                generated_original_image = dlgips_image_generator(geomery_features, texture_features)  # G(f_src_g,f_t)

                # 将 生成的投影  再放入 feature encoder中，得到 features
                generated_ap_feature = dlgips_geometry_encoder(generated_image_ap)  # ε_g(I_src')
                generated_lt_feature = dlgips_texture_encoder(generated_image_lt)  # ε_g(I_tgt')

                loss_cyc = consistency_loss.total_consistency_loss(generated_original_image,
                                                                   ap_truth, generated_lt_feature,
                                                                   transformed_features_lt,
                                                                   generated_ap_feature, transformed_features_ap)
                loss_rec = reconstruction_loss.reconstruction_loss(generated_image_lt, lt_truth,
                                                                   generated_image_ap,
                                                                   ap_truth)
                val_loss = cyc_weight * loss_cyc + rec_weight * loss_rec

                val_loss_total += val_loss.item()

            avg_val_loss = val_loss_total / len(val_loader)
            # print(f'Epoch {epoch}, Average Validation Loss: {avg_val_loss}')


            # Log epoch summary
        with open(log_path, 'a') as log_file:
            log_file.write(f'{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f}\n')

        print(f'Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')

        # Save checkpoint 每隔20个保存一个checkpoints
        if epoch % 20 == 0:
            checkpoint = {
                'epoch': epoch,
                'geometry_encoder_state_dict': dlgips_geometry_encoder.state_dict(),
                'texture_encoder_state_dict': dlgips_texture_encoder.state_dict(),
                'geometry_transform_state_dict': dlgips_geometry_transform.state_dict(),
                'image_generator_state_dict': dlgips_image_generator.state_dict(),
                'optimizer_geometry_encoder': optimizer_geometry_encoder.state_dict(),
                'optimizer_texture_encoder': optimizer_texture_encoder.state_dict(),
                'optimizer_geometry_transform': optimizer_geometry_transform.state_dict(),
                'optimizer_image_generator': optimizer_image_generator.state_dict(),
                'loss': avg_val_loss,
                # 如果还有其他需要保存的信息，可以继续添加
            }
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch}')


        epoch += 1