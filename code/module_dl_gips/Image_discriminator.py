'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    """
    单独的分类器网络，用于处理不同尺度的图像
    从其输入图像中提取特征
    """
    def __init__(self, input_channels):
        super(Classifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, generated_image):
        return self.conv_layers(generated_image)  # 经过卷积层得到的特征值


class ImageDiscriminator(nn.Module):
    """多尺度图像鉴别器"""
    def __init__(self, input_channels):
        super(ImageDiscriminator, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.classifier_original = Classifier(input_channels)
        self.classifier_scale_1 = Classifier(input_channels)
        self.classifier_scale_2 = Classifier(input_channels)

# def forward(self, generated_image):
#         output_original = self.classifier_original(generated_image)
#         x_scaled_1 = self.pool(generated_image)
#         output_scaled_1 = self.classifier_scale_1(x_scaled_1)
#         x_scaled_2 = self.pool(x_scaled_1)
#         output_scaled_2 = self.classifier_scale_2(x_scaled_2)
#         return output_original, output_scaled_1, output_scaled_2


    def forward(self, generated_image):
        # 应用Sigmoid并计算每个尺度的输出
        # Sigmoid函数将输入压缩到0和1之间，生成概率似的输出。
        #  generated_image（1，1，300，180） 经过卷积层得到的特征值
        output_original = torch.sigmoid(self.classifier_original(generated_image))

        # （1，1，150，90）经过卷积层得到的特征值
        x_scaled_1 = self.pool(generated_image)
        output_scaled_1 = torch.sigmoid(self.classifier_scale_1(x_scaled_1))
        # （1，1，75，45）经过卷积层得到的特征值
        x_scaled_2 = self.pool(x_scaled_1)
        output_scaled_2 = torch.sigmoid(self.classifier_scale_2(x_scaled_2))

        # 计算平均评分
        averaged_output = (output_original + output_scaled_1 + output_scaled_2) / 3
        return averaged_output
'''




'''
我自己的
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageDiscriminator(nn.Module):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
        self.scale1 = DiscriminatorScale()
        self.scale2 = DiscriminatorScale()
        self.scale3 = DiscriminatorScale()
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        out_scale1 = self.scale1(x)
        x_downsampled = self.avg_pool(x)
        out_scale2 = self.scale2(x_downsampled)
        x_downsampled = self.avg_pool(x_downsampled)
        out_scale3 = self.scale3(x_downsampled)
        return out_scale1, out_scale2, out_scale3

class DiscriminatorScale(nn.Module):
    def __init__(self):
        super(DiscriminatorScale, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.final_conv(x)
        return x


# gpt 给的
