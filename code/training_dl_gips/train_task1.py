
import code.dataset_prepare.projection_dataset
import numpy as np
import reconstruction_loss
import consistency_loss
import adversarial_loss
import torch
import torch.nn as nn
import code.module_dl_gips.segmen_task1
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
from PIL import Image
from torchvision.transforms import ToPILImage
import tifffile as tiff


# 定义一个转换来将tensor转换回图像格式
# 这个转换将标准化过的图像张量转换回可显示的图像格式


"""
首先一开始的data中要包含 ap lt两个角度的projection，尽管进入网络的只有ap方向的projection
"""
# train AP_AP&LT

dataset = code.dataset_prepare.projection_dataset.ProjectionTensorDataset(ap_dir='G:/dataset/projection_ap', lt_dir='G:/dataset/projection_lt')
dataloader = dataset.DataLoader(dataset, batch_size=1, shuffle=True)

input_channel = 1
output_channel = 1

# Define a directory to save the TIF images
tif_output_dir = os.path.join(root_dir, 'tif_output')
os.makedirs(tif_output_dir, exist_ok=True)


# input_channel=1 ,output_channel=1
dlgips_geometry_encoder=segmen_task1.DLGIPS_Geometry_encoder(input_channel,output_channel)
dlgips_texture_encoder=segmen_task1.DLGIPS_Texture_encoder(input_channel,output_channel)
dlgips_geometry_transform=segmen_task1.DLGIPS_geometry_transform(input_channel, target_shape, proj_size,
                                                                 num_proj_ap, start_angle_ap, end_angle_ap,
                                                                 num_proj_aplt, start_angle_aplt, end_angle_aplt)
# geometry_channels=texture_channels=output_channel=1
dlgips_image_generator=segmen_task1.DLGIPS_image_generator(output_channel,output_channel)

dlgips_image_discriminator=segmen_task1.DLGIPS_image_discriminator()

device = torch.device("cuda")
dlgips_geometry_encoder = dlgips_geometry_encoder.cuda()    # ε_g
dlgips_texture_encoder = dlgips_texture_encoder.cuda()      # ε_t
dlgips_geometry_transform = dlgips_geometry_transform.cuda()  # BP&Net&FP
dlgips_image_generator = dlgips_image_generator.cuda()     # G
dlgips_image_discriminator = dlgips_image_discriminator.cuda()  # D


num_epochs=1
batch_size = 5

# 定义优化器和超参数
optimizer_geometry_encoder = torch.optim.Adam(dlgips_geometry_encoder.parameters(), lr=0.0001)
optimizer_texture_encoder = torch.optim.Adam(dlgips_texture_encoder.parameters(), lr=0.0001)

optimizer_geometry_transform = torch.optim.Adam(dlgips_geometry_transform.parameters(), lr=0.0001)
optimizer_image_generator = torch.optim.Adam(dlgips_image_generator.parameters(), lr=0.0001)
optimizer_image_discriminator = torch.optim.Adam(dlgips_image_discriminator.parameters(), lr=0.0001)
cyc_weight, rec_weight, adv_weight = 1, 10, 1


torch.autograd.set_detect_anomaly(True)

# Initialize loss history

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i, data in enumerate(projection_dataloader):

        # projection_data_gpu : ap角度的projection
        projection_data_gpu = data.to(device)
        # print("projection_data_gpu.shape:",projection_data_gpu.shape)

        # 得到的是（1，1，2，300，180） 去除一个维度
        reduced_tensor = projection_data_gpu.squeeze(1)
        # projection_data_gpu 是 （1，2，300，180）的ground truth
        ap_truth = reduced_tensor[:, 0:1, :, :]  # ap投影，形状为 (1, 1, 300, 180)   # I_src
        lt_truth = reduced_tensor[:, 1:2, :, :]  # lt投影，形状为 (1, 1, 300, 180)   # I_tgt
        # print("ap_truth.shape:",ap_truth.shape)

        geomery_features=dlgips_geometry_encoder(ap_truth)  # f_src_g
        texture_features=dlgips_texture_encoder(ap_truth)   # f_t

        # F_src_g                   F_tgt_g
        transformed_features_ap, transformed_features_lt=dlgips_geometry_transform(geomery_features)

        generated_image_ap=dlgips_image_generator(transformed_features_ap,texture_features) # I_src'
        generated_image_lt=dlgips_image_generator(transformed_features_lt,texture_features) # I_tgt'
        generated_original_image=dlgips_image_generator(geomery_features,texture_features)  # G(f_src_g,f_t)

        out_scale1_ap, out_scale2_ap, out_scale3_ap=dlgips_image_discriminator(generated_image_ap)
        out_scale1_lt, out_scale2_lt, out_scale3_lt=dlgips_image_discriminator(generated_image_lt)

        # 将 生成的投影  再放入 feature encoder中，得到 features
        generated_ap_feature=dlgips_geometry_encoder(generated_image_ap) #ε_g(I_src')
        generated_lt_feature=dlgips_texture_encoder(generated_image_lt)  #ε_g(I_tgt')




        # ground truth ---> ap_truth,lt_truth

        # 计算损失
        # print("generated_image_lt.shape:",generated_image_lt.shape)
        # print("lt_truth:",lt_truth.shape)
        loss_cyc = consistency_loss.total_consistency_loss(generated_original_image,
                                                           ap_truth, generated_lt_feature,
                                                           transformed_features_lt,
                                                           generated_ap_feature, transformed_features_ap)
        loss_rec = reconstruction_loss.reconstruction_loss(generated_image_lt, lt_truth,
                                                           generated_image_ap,
                                                           ap_truth)
        # loss_adv = adversarial_loss.adversarial_loss(dlgips_image_discriminator, generated_image_lt
        #                                              , lt_truth,device)

        # 组合总损失
        optimizer_geometry_encoder.zero_grad()
        optimizer_texture_encoder.zero_grad()
        optimizer_geometry_transform.zero_grad()
        optimizer_image_generator.zero_grad()
        optimizer_image_discriminator.zero_grad()
        total_loss = cyc_weight * loss_cyc + rec_weight * loss_rec

        print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {total_loss.item()}')

        # + adv_weight * loss_adv
# Backpropagation for feature encoder, geometry transform, and image generator
        total_loss.backward(retain_graph=True)
        optimizer_geometry_encoder.step()
        optimizer_texture_encoder.step()
        optimizer_geometry_transform.step()
        optimizer_image_generator.step()
    generated_image_ap_numpy = generated_image_ap.cpu().detach().numpy().squeeze()
    tif_filename = f'generated_image_ap_epoch_{epoch + 1}.tif'
    tif_filepath = os.path.join(tif_output_dir, tif_filename)
    tiff.imsave(tif_filepath, generated_image_ap_numpy)
    print(f'Saved generated image as TIF at: {tif_filepath}')

    # For discriminator, only use adversarial loss for backpropagation
        # loss_adv.backward()
        # optimizer_image_discriminator.step()


torch.cuda.empty_cache()
# 训练结束后清除 CUDA 缓存
torch.cuda.empty_cache()
