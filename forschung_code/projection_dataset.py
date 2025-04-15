


import os
import torch
# from torch.utils.data import Dataset, DataLoader
# import pydicom
# import numpy as np
# import volume_padding
# import code.different_task.task1
# import tifffile as tiff
'''
class ProjectionDataset(Dataset):
    def __init__(self, root_dir, target_shape,proj_size, num_proj,start_angle, end_angle):
        self.root_dir = root_dir
        self.target_shape = target_shape  #(500,128,128)
        self.proj_size = proj_size   # (300,180)
        self.num_proj=num_proj
        self.start_angle = start_angle
        self.end_angle = end_angle
        # patients 里面是所有子文件（G:/dataset/LIDC-IDRI\LIDC-IDRI-0001）的路径
        self.patients = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_folder = self.patients[idx]
        # 得到的是 numpy (500,128,128) idx病人的volume
        volume = volume_padding.symmetric_padding_3d_volume(patient_folder, self.target_shape)
        # 将NumPy数组转换为PyTorch张量，并添加批次和通道维度
        # Save the volume as a tif file
        # tiff.imsave(f'dataloader_volume_{idx}.tif', volume.astype(np.uint16))  # Ensure the correct data type

        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        # 对于task1来说，得到的dataloader是 ap 和 lt 两个角度的 projection
        ap_aplt = task1.AP_APLT_Operators(self.target_shape, self.proj_size, self.num_proj, self.start_angle, self.end_angle)
        projection = ap_aplt.forward_project(volume_tensor)
        return projection
    # 此时的projection 是 tensor.shape=（1，2，300，180）
'''
ap_dir = 'G:/dataset/projection_ap'
lt_dir = 'G:/dataset/projection_lt'

from torch.utils.data import Dataset, DataLoader

class ProjectionTensorDataset_ap(Dataset):
    def __init__(self, ap_dir):
        self.ap_files = [os.path.join(ap_dir, file) for file in os.listdir(ap_dir)]


    def __len__(self):
        return len(self.ap_files)

    def __getitem__(self, idx):
        ap_projection = torch.load(self.ap_files[idx])

        return ap_projection

class ProjectionTensorDataset_lt(Dataset):
    def __init__(self,lt_dir):

        self.lt_files = [os.path.join(lt_dir, file) for file in os.listdir(lt_dir)]

    def __len__(self):
        return len(self.lt_files)

    def __getitem__(self, idx):

        lt_projection = torch.load(self.lt_files[idx])
        return lt_projection

class ProjectionTensorDataset(Dataset):
    def __init__(self, ap_dir,lt_dir):
        self.ap_files = [os.path.join(ap_dir, file) for file in os.listdir(ap_dir)]
        self.lt_files = [os.path.join(lt_dir, file) for file in os.listdir(lt_dir)]

    def __len__(self):
        return len(self.ap_files)

    def __getitem__(self, idx):
        ap_projection = torch.load(self.ap_files[idx])
        lt_projection = torch.load(self.lt_files[idx])
        return ap_projection,lt_projection
#
# # 实例化数据集
# dataset = ProjectionTensorDataset_ap(ap_dir)
#
# # 遍历数据集并打印形状
# for i in range(len(dataset)):
#     ap_projection = dataset[i]
#     print(f"Index {i}: AP projection shape: {ap_projection.shape}")