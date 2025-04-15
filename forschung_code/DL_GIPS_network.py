
import torch
import torch.nn as nn

import texture_encoder
import geometry_encoder
import Image_Generator
import Image_discriminator
import Geometry_transformation
import numpy as np





class DLGIPS(nn.Module):
    def __init__(self,input_channel,output_channel,image_size, proj_size,
                 num_proj_initialization,start_angle_initialization,end_angle_initialization,
                 num_proj_transform,start_angle_transform,end_angle_transform):
        super(DLGIPS, self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel

        self.image_size=image_size
        self.proj_size=proj_size
        self.num_proj_initialization=num_proj_initialization
        self.start_angle_initialization=start_angle_initialization
        self.end_angle_initialization=end_angle_initialization
        self.num_proj_transform=num_proj_transform
        self.start_angle_transform=start_angle_transform
        self.end_angle_transform=end_angle_transform

        self.geometry_encoder = geometry_encoder.Geometry_Encoder(self.input_channel,self.output_channel)
        self.texture_encoder = texture_encoder.Texture_Encoder(self.input_channel,self.output_channel)
        self.geometry_transformation = Geometry_transformation.GeometryTransformation(self.output_channel,self.image_size,self.proj_size,self.num_proj_initialization,self.start_angle_initialization,self.end_angle_initialization,self.num_proj_transform,self.start_angle_transform,self.end_angle_transform)
        self.image_generator = Image_Generator.ImageGenerator(self.output_channel,self.output_channel)


    def forward(self, projcetion_tensor):
        # feature_encoder
        geometry_features = self.geometry_encoder(projcetion_tensor)
        texture_features=self.texture_encoder(projcetion_tensor)
        # geometry_transform
        transformed_features = self.geometry_transformation(geometry_features)

        transformed_features_ap, transformed_features_lt = torch.chunk(transformed_features, 2, dim=1)
        generated_image_ap = self.image_generator(transformed_features_ap, texture_features)
        generated_image_lt = self.image_generator(transformed_features_lt, texture_features)
        return generated_image_ap,generated_image_lt

if __name__ == '__main__':
    x=torch.randn(1,1,300,180)


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
    net= DLGIPS(input_channel,output_channel,image_size,proj_size,num_proj_ap,start_angle_ap,end_angle_ap,num_proj_aplt,start_angle_aplt,end_angle_aplt)
    print(net(x)[1].shape)