import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import odl
from odl.contrib import torch as odl_torch
import ct_geometry_projector
import projection_dataset
import tifffile as tiff
import projection_dataset


import os
import tifffile
from torch.utils.data import DataLoader

"""
输入是ap方向的投影，调用的是AP_AP.backward，从而得到volume
得到的volume经过refinment model,进行补全后
调用的是AP_APLT.forward,从而得到ap和lt方向的projcetion
"""


#
# #  1 --> 1
# class AP_AP_Operators():
#     def __init__(self, image_size, proj_size, num_proj, start_angle, end_angle):
#
#         self.image_size = image_size
#         self.proj_size = proj_size
#         self.num_proj = num_proj
#         self.start_angle = start_angle
#         self.end_angle = end_angle
#         self.raw_reso = 2.8
#
#         # Initialize required parameters for image, view, detector
#         geo_param = ct_geometry_projector.Initialization_ConeBeam(image_size=self.image_size,
#                                                                   num_proj=self.num_proj,
#                                                                   start_angle=self.start_angle,
#                                                                   end_angle=self.end_angle,
#                                                                   proj_size=self.proj_size,
#                                                                   raw_reso=self.raw_reso)
#
#         # Initialize forward and backward projectors for both AP and LT.
#         self.forward_projector = ct_geometry_projector.Projection_ConeBeam(geo_param)
#         self.backward_projector = ct_geometry_projector.FBP_ConeBeam(geo_param)
#     def backward_project(self, projection_data_ap):
#         """
#         Backward Projection Operator:
#         projection_data_ap  ：tensor  (1,1,300,180)
#         要先转化成 （1，1，1，300，180）的tensor
#          return : (1,1,500,128,128)
#         """
#         expanded_tensor = projection_data_ap.unsqueeze(1)
#         volume = self.backward_projector(expanded_tensor)
#         print(f" chuan ru ap : {projection_data_ap.shape}")
#         print(f" expanded_tensor: {expanded_tensor.shape}")
#         print(f" Tensor AP_AP Backward  3D volume shape: {volume.shape}")
#         return volume
#
#     def forward_project(self, volume):
#         """
#         Forward Projection for AP-LT Transformation:
#         Computes 2D projections at 0 degrees (AP) and 90 degrees (LT) from a 3D volume.
#         输入的volume是（1，1，500，128，128）
#         得到的 proj 是（1，1，1，300，180）
#         然后将 得到的 proj进行 去除维度，变成 （1，1，300，180），方便进入后续的神经网络
#         """
#         projections_ap_ap = self.forward_projector(volume)
#         reduced_tensor = projections_ap_ap.squeeze(1)
#         print(f" Tensor Volume_Forward projection AP shape: {reduced_tensor.shape}")
#         return reduced_tensor
#
# #  1 --> 2
# class AP_APLT_Operators():
#     def __init__(self, image_size, proj_size, num_proj,start_angle,end_angle):
#
#         self.image_size = image_size
#         self.proj_size = proj_size
#         self.num_proj = num_proj
#         self.start_angle=start_angle
#         self.end_angle=end_angle
#         self.raw_reso = 2.8
#
#         # Initialize required parameters for image, view, detector
#         geo_param = ct_geometry_projector.Initialization_ConeBeam(image_size=self.image_size,
#                                             num_proj=self.num_proj,
#                                             start_angle=self.start_angle,
#                                             end_angle=self.end_angle,
#                                             proj_size=self.proj_size,
#                                             raw_reso=self.raw_reso)
#
#
#         # Initialize forward and backward projectors for both AP and LT.
#         self.forward_projector = ct_geometry_projector.Projection_ConeBeam(geo_param)
#         self.backward_projector = ct_geometry_projector.FBP_ConeBeam(geo_param)
#
#     def backward_project(self, projection_data_ap):
#         """
#         Backward Projection Operator:
#         projection_data_ap  ：tensor  (1,1,300,180)
#         要先转化成 （1，1，1，300，180）的tensor
#         得到的
#               """
#         expanded_tensor = projection_data_ap.unsqueeze(1)
#         volume = self.backward_projector(expanded_tensor)
#         # print(f" Tensor AP_AP&LT Backward  3D volume shape: {volume.shape}")
#         return volume
#
#
#     def forward_project(self, volume):
#         """
#         Forward Projection for AP-LT Transformation:
#         Computes 2D projections at 0 degrees (AP) and 90 degrees (LT) from a 3D volume.
#         输入的volume是（1，1，500，128，128）
#         得到的 proj 是（1，1，2，300，180）
#         然后将 得到的 proj进行 去除维度，变成 （1，2，300，180），方便进入后续的神经网络
#         """
#         projections_ap_ap=self.forward_projector(volume)
#         reduced_tensor = projections_ap_ap.squeeze(1)
#         # print(f" Tensor Volume_Forward projection AP&LT shape: {reduced_tensor.shape}")
#         return reduced_tensor
#

#  1 --> 1
class AP_AP_Operators():
    def __init__(self, image_size, proj_size, num_proj, start_angle, end_angle):

        self.image_size = image_size
        self.proj_size = proj_size
        self.num_proj = num_proj
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.raw_reso = 2.8

        # Initialize required parameters for image, view, detector
        geo_param = ct_geometry_projector.Initialization_ConeBeam(image_size=self.image_size,
                                                                  num_proj=self.num_proj,
                                                                  start_angle=self.start_angle,
                                                                  end_angle=self.end_angle,
                                                                  proj_size=self.proj_size,
                                                                  raw_reso=self.raw_reso)

        # Initialize forward and backward projectors for both AP and LT.
        self.forward_projector = ct_geometry_projector.Projection_ConeBeam(geo_param)
        self.backward_projector = ct_geometry_projector.FBP_ConeBeam(geo_param)
    def backward_project(self, projection_data_ap):
        """
        Backward Projection Operator:
        projection_data_ap  ：tensor  (1,1,300,180)
        要先转化成 （1，1，1，300，180）的tensor
         return : (1,1,500,128,128)
        """

        volume = self.backward_projector(projection_data_ap)
        print(f" chuan ru ap : {projection_data_ap.shape}")

        print(f" Tensor AP_AP Backward  3D volume shape: {volume.shape}")
        return volume

    def forward_project(self, volume):
        """
        Forward Projection for AP-LT Transformation:
        Computes 2D projections at 0 degrees (AP) and 90 degrees (LT) from a 3D volume.
        输入的volume是（1，1，500，128，128）
        得到的 proj 是（1，1，1，300，180）
        然后将 得到的 proj进行 去除维度，变成 （1，1，300，180），方便进入后续的神经网络
        """
        projections_ap_ap = self.forward_projector(volume)

        print(f" Tensor Volume_Forward projection AP shape: {projections_ap_ap.shape}")
        return projections_ap_ap

#  1 --> 2
class AP_APLT_Operators():
    def __init__(self, image_size, proj_size, num_proj,start_angle,end_angle):

        self.image_size = image_size
        self.proj_size = proj_size
        self.num_proj = num_proj
        self.start_angle=start_angle
        self.end_angle=end_angle
        self.raw_reso = 2.8

        # Initialize required parameters for image, view, detector
        geo_param = ct_geometry_projector.Initialization_ConeBeam(image_size=self.image_size,
                                            num_proj=self.num_proj,
                                            start_angle=self.start_angle,
                                            end_angle=self.end_angle,
                                            proj_size=self.proj_size,
                                            raw_reso=self.raw_reso)


        # Initialize forward and backward projectors for both AP and LT.
        self.forward_projector = ct_geometry_projector.Projection_ConeBeam(geo_param)
        self.backward_projector = ct_geometry_projector.FBP_ConeBeam(geo_param)

    def backward_project(self, projection_data_ap):
        """
        Backward Projection Operator:
        projection_data_ap  ：tensor  (1,1,300,180)
        要先转化成 （1，1，1，300，180）的tensor
        得到的
              """

        volume = self.backward_projector(projection_data_ap)
        # print(f" Tensor AP_AP&LT Backward  3D volume shape: {volume.shape}")
        return volume


    def forward_project(self, volume):
        """
        Forward Projection for AP-LT Transformation:
        Computes 2D projections at 0 degrees (AP) and 90 degrees (LT) from a 3D volume.
        输入的volume是（1，1，500，128，128）
        得到的 proj 是（1，1，2，300，180）
        然后将 得到的 proj进行 去除维度，变成 （1，2，300，180），方便进入后续的神经网络
        """
        projections_ap_ap=self.forward_projector(volume)

        # print(f" Tensor Volume_Forward projection AP&LT shape: {reduced_tensor.shape}")
        return projections_ap_ap


if __name__ == '__main__':
    # Set the directories and parameters
    ap_dir = 'G:/dataset/projection_ap'
    lt_dir = 'G:/dataset/projection_lt'
    output_dir = 'E:/Fau/ws2023/Forschung/check/check_task1'
    os.makedirs(output_dir, exist_ok=True)

    # Dataset and DataLoader setup
    dataset = projection_dataset.ProjectionTensorDataset(ap_dir, lt_dir)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Parameters for the operators
    image_size = (500, 128, 128)
    proj_size = (300, 180)
    num_proj = 1
    start_angle = -np.pi / 2
    end_angle = np.pi / 2
    num_proj2 = 2
    start_angle2 = -np.pi / 2
    end_angle2 = np.pi

    # Initialize the operators
    ap_ap_operators = AP_AP_Operators(image_size=image_size, proj_size=proj_size,
                                      num_proj=num_proj, start_angle=start_angle,
                                      end_angle=end_angle)
    ap_aplt_operators = AP_APLT_Operators(image_size=image_size, proj_size=proj_size,
                                          num_proj=num_proj2, start_angle=start_angle2,
                                          end_angle=end_angle2)

    # Process the first patient's AP projection
    first_ap_projection, _ = next(iter(dataloader))
    first_ap_projection = first_ap_projection.to(torch.float32)  # Ensuring the data type matches

    # Generate 3D volume from AP projection
    volume = ap_ap_operators.backward_project(first_ap_projection)
    # 仅check ap_ap
    ap_check=ap_ap_operators.forward_project(volume)
    ap_check=ap_check.squeeze(0).numpy()
    print("zui hou bao cun :::",ap_check.shape)
    tifffile.imsave(os.path.join(output_dir, 'ap_ap_projection.tif'), ap_check)

    # Generate AP and LT projections from the volume
    ap_lt_projections = ap_aplt_operators.forward_project(volume)
    print("dudud",ap_lt_projections.shape)
    # Save the resulting AP and LT projections
    ap_projection = ap_lt_projections[0, 0, :, :].numpy()  # Assuming the correct index
    lt_projection = ap_lt_projections[0, 1, :, :].numpy()  # Assuming the correct index

    print("zui hou bao cun",ap_projection.shape)

    # Save to TIFF
    tifffile.imsave(os.path.join(output_dir, 'patient1_ap_projection.tif'), ap_projection)
    tifffile.imsave(os.path.join(output_dir, 'patient1_lt_projection.tif'), lt_projection)
    print("Saved AP and LT projections for the first patient.")

