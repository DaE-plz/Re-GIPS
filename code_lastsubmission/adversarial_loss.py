'''
对抗损失（Adversarial Loss）
用于强化生成投影与真实投影的分布接近，使得生成的投影更加逼真。
采用的是最小二乘生成对抗网络（Least Squares Generative Adversarial Networks，LSGANs）
的目标损失函数，
其中 a (0)和 b(1) 是合成图像和真实图像的标签。
'''

import torch

def adversarial_loss(disc_real_outputs, disc_generated_outputs, real_label, fake_label):
    # 初始化损失函数
    loss_fn = torch.nn.MSELoss()
    real_loss = loss_fn(disc_real_outputs, real_label)
    fake_loss = loss_fn(disc_generated_outputs, fake_label)
    return real_loss+fake_loss