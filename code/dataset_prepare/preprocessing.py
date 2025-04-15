import os
import pydicom
import numpy as np
from scipy.ndimage import zoom
import loaddcm

def preprocess_dicom_images(dicom_folder):
    original_volume = loaddcm.stack_dicom_files(dicom_folder)
    if original_volume is None:
        print("无法加载 DICOM 文件或文件夹为空")
        return None, None, None, None

    # 计算原始像素间距
    first_dicom_file = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')][0]
    first_ds = pydicom.dcmread(first_dicom_file)
    original_pixel_spacing_x, original_pixel_spacing_y = map(float, first_ds.PixelSpacing)
    slice_shape = original_volume.shape[1:]  # 获取原始体积的 XY 尺寸

    # 重新采样 z 轴和调整 xy 平面尺寸
    new_size_xy = (128, 128)
    resampled_slice_thickness = 1.0  # mm
    z_factor = float(first_ds.SliceThickness) / resampled_slice_thickness
    x_factor = new_size_xy[0] / slice_shape[1]
    y_factor = new_size_xy[1] / slice_shape[0]
    preprocessed_volume = zoom(original_volume, (z_factor, x_factor, y_factor), order=1)



    # 计算新的像素间距
    new_pixel_spacing_x = original_pixel_spacing_x / x_factor
    new_pixel_spacing_y = original_pixel_spacing_y / y_factor

    return preprocessed_volume, new_pixel_spacing_x, new_pixel_spacing_y, resampled_slice_thickness




'''

# 示例用法
dicom_folder = 'G:/dataset/LIDC-IDRI/LIDC-IDRI-0360'
preprocessed_volume, new_pixel_spacing_x, new_pixel_spacing_y, resampled_slice_thickness = preprocess_dicom_images(dicom_folder)

if preprocessed_volume is not None:
    print("预处理后的3D体积形状:", preprocessed_volume.shape)
    print(f"新的像素空间分辨率 (X): {new_pixel_spacing_x} mm/pixel")
    print(f"新的像素空间分辨率 (Y): {new_pixel_spacing_y} mm/pixel")
    print(f"Z轴重新采样后的分辨率: {resampled_slice_thickness} mm")

# 保存预处理后的体积
output_file = 'preprocessed_volume_0360.tif'
# 这里假定 save_3d_volume_as_tif 是一个有效的函数，用于保存 TIFF 文件
loaddcm.save_3d_volume_as_tif(preprocessed_volume, output_file)
'''
