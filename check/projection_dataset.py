


import os
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import numpy as np
import tifffile

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
    def __init__(self, ap_dir, lt_dir):
        self.ap_files = sorted([os.path.join(ap_dir, file) for file in os.listdir(ap_dir)])
        self.lt_files = sorted([os.path.join(lt_dir, file) for file in os.listdir(lt_dir)])
        assert len(self.ap_files) == len(self.lt_files), "Mismatch in dataset size between AP and LT files."

    def __len__(self):
        return len(self.ap_files)

    def __getitem__(self, idx):
        try:
            ap_projection = torch.load(self.ap_files[idx])
            lt_projection = torch.load(self.lt_files[idx])
        except Exception as e:
            print(f"Failed to load files {self.ap_files[idx]} or {self.lt_files[idx]}: {str(e)}")
            return None, None
        return ap_projection,lt_projection


if __name__ == "__main__":
    ap_dir = 'G:/dataset/projection_ap'
    lt_dir = 'G:/dataset/projection_lt'

    # Initialize dataset
    dataset = ProjectionTensorDataset(ap_dir, lt_dir)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Define the output directory
    output_dir = 'E:/Fau/ws2023/Forschung/check'
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it does not exist

    # Iterate over the dataloader to fetch the projections
    for i, (ap_projection, lt_projection) in enumerate(dataloader):
        print(ap_projection.shape)
        # if i >= 3:  # Stop after saving the first three patients' data
        #     break
        # # Define output file paths
        # output_ap_file = os.path.join(output_dir, f'ap_patient_{i + 1}.tif')
        # output_lt_file = os.path.join(output_dir, f'lt_patient_{i + 1}.tif')
        #
        # # Save AP and LT projections as TIFF files
        # tifffile.imsave(output_ap_file, ap_projection.squeeze(0).numpy())
        # tifffile.imsave(output_lt_file, lt_projection.squeeze(0).numpy())
        #
        # print(f"Saved AP and LT projections for patient {i + 1} as TIFF files.")