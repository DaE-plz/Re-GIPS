'''
对抗损失（Adversarial Loss）
用于强化生成投影与真实投影的分布接近，使得生成的投影更加逼真。
采用的是最小二乘生成对抗网络（Least Squares Generative Adversarial Networks，LSGANs）
的目标损失函数，
其中 a (0)和 b(1) 是合成图像和真实图像的标签。
'''

import torch.nn.functional as F
import torch
#
# def adversarial_loss(discriminator, generated_image, real_image, real_label, fake_label):
#     """
#     对抗损失
#     :param discriminator: 鉴别器模型
#     :param generated_image: 由生成器生成的图像
#     :param real_image: 真实图像
#     :param real_label: 真实图像的标签
#     :param fake_label: 合成图像的标签
#     :return: 损失值
#     """
#
#     # 鉴别器对合成图像的预测
#     pred_fake = discriminator(generated_image)
#     # 鉴别器对真实图像的预测
#     pred_real = discriminator(real_image)
#
#     # 对抗损失（LSGAN）
#     loss_fake = F.mse_loss(pred_fake, fake_label)
#     loss_real = F.mse_loss(pred_real, real_label)
#     return (loss_fake + loss_real) / 2

# def adversarial_loss(discriminator, generated_image, real_image, device):
#
#     # Generate labels for fake and real images
#     # Assuming pred_fake and pred_real are single-dimensional outputs
#     fake_label = torch.zeros_like(discriminator(generated_image))
#     real_label = torch.ones_like(discriminator(real_image))
#
#     # Move labels to the correct device
#     fake_label = fake_label.to(device)
#     real_label = real_label.to(device)
#
#     # Discriminator output for fake and real images
#     pred_fake = discriminator(generated_image)
#     pred_real = discriminator(real_image)
#
#     # Calculate loss
#     loss_fake = F.mse_loss(pred_fake, fake_label)
#     loss_real = F.mse_loss(pred_real, real_label)
#     return (loss_fake + loss_real) / 2



# 对discriminator的三个尺度的输出 分别进行评判
def adversarial_loss(discriminator, generated_image, real_image, device):
    """
          对抗损失
          :param discriminator: 鉴别器模型
          :param generated_image: 由生成器生成的图像
          :param real_image: 真实图像
          :param real_label: 真实图像的标签
          :param fake_label: 合成图像的标签
          :return: 损失值
          """
    # Discriminator output for fake and real images
    pred_fake_scales = discriminator(generated_image)
    pred_real_scales = discriminator(real_image)

    loss_fake_total = 0.0
    loss_real_total = 0.0

    for pred_fake, pred_real in zip(pred_fake_scales, pred_real_scales):
        # Generate labels for fake and real images
        fake_label = torch.zeros_like(pred_fake).to(device)
        real_label = torch.ones_like(pred_real).to(device)

        # Calculate loss for this scale
        loss_fake = F.mse_loss(pred_fake, fake_label)
        loss_real = F.mse_loss(pred_real, real_label)

        loss_fake_total += loss_fake
        loss_real_total += loss_real

    # Average the losses across all scales
    num_scales = len(pred_fake_scales)
    return (loss_fake_total + loss_real_total) / (2 * num_scales)