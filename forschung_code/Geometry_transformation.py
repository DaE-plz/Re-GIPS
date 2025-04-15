import torch
import torch.nn as nn
import torch.nn.functional as F
import ct_geometry_projector
import task1
import numpy as np

class ResidualBlock3D(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += identity
        return F.relu(out)

class GeometryTransformation(nn.Module):
    def __init__(self, channels, image_size, proj_size,
                 num_proj_ap, start_angle_ap, end_angle_ap,
                 num_proj_aplt, start_angle_aplt, end_angle_aplt):
        super(GeometryTransformation, self).__init__()
        self.channels=channels
        self.image_size = image_size
        self.proj_size = proj_size
        self.num_proj_ap = num_proj_ap
        self.start_angle_ap = start_angle_ap
        self.end_angle_ap = end_angle_ap
        self.num_proj_aplt=num_proj_aplt
        self.start_angle_aplt=start_angle_aplt
        self.end_angle_aplt=end_angle_aplt

        self.ap_ap=task1.AP_AP_Operators(self.image_size, self.proj_size, self.num_proj_ap, self.start_angle_ap, self.end_angle_ap)
        self.ap_aplt=task1.AP_APLT_Operators(self.image_size, self.proj_size, self.num_proj_aplt, self.start_angle_aplt, self.end_angle_aplt)

        # 定义3D refinement model
        self.refinement_model = nn.Sequential(
            # 两个3D残差卷积块
            ResidualBlock3D(self.channels),
            ResidualBlock3D(self.channels)
        )


    def backward_projection(self, projection_data):

        return self.ap_ap.backward_project(projection_data)

    def forward_projection(self, volume):

        return self.ap_aplt.forward_project(volume)

    def forward(self, projection_data):
        # backward
        volume = self.backward_projection(projection_data)
        # 3D  refined
        refined_volume = self.refinement_model(volume)

        # forward
        # 对于AP_AP 生成的是（1，1，300，180）
        # 对于AP_AP&LT 生成的是（1，2，300，180）
        forward_projected = self.forward_projection(refined_volume)

        return forward_projected



# input:geometry_features (1,1,300,180)  --->  output:transformed_features (1,2,300,180)

if __name__ == '__main__':
    x=torch.randn(1,1,300,180)

    channels=1
    image_size = [500, 128, 128]
    proj_size= [300, 180]

    num_proj_ap=1
    start_angle_ap= -np.pi /2
    end_angle_ap= np.pi /2

    num_proj_aplt=2
    start_angle_aplt= -np.pi /2
    end_angle_aplt= np.pi

    net= GeometryTransformation(channels,image_size,proj_size,num_proj_ap,start_angle_ap,end_angle_ap,num_proj_aplt,start_angle_aplt,end_angle_aplt
    )
    print(net(x).shape)