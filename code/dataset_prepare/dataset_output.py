import os
import numpy as np
import torch
import volume_padding
import code.different_task.task1
import tifffile as tiff



root_dir = 'G:/dataset/LIDC-IDRI'
target_shape = (500, 128, 128)
proj_size = [300, 180]
num_proj = 2
start_angle = -np.pi / 2
end_angle = np.pi


# 创建保存投影的目录
ap_dir = 'G:/dataset/projection_ap'
lt_dir = 'G:/dataset/projection_lt'
os.makedirs(ap_dir, exist_ok=True)
os.makedirs(lt_dir, exist_ok=True)

# 初始化task1的operator
aplt_operator = code.different_task.task1.AP_APLT_Operators(target_shape, proj_size, num_proj, start_angle, end_angle)


for idx, patient in enumerate(os.listdir(root_dir)):
    patient_folder = os.path.join(root_dir, patient)
    if os.path.isdir(patient_folder):
        # 应用volume_padding获取3D体积
        volume = volume_padding.symmetric_padding_3d_volume(patient_folder, target_shape)
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()

        projection = aplt_operator.forward_project(volume_tensor)

        # 提取AP和LT投影
        ap_projection = projection[:, 0, :, :]  # AP投影在第二维的第一个位置
        lt_projection = projection[:, 1, :, :]  # LT投影在第二维的第二个位置

        # 保存AP和LT投影
        tiff.imsave(os.path.join(ap_dir, f'{patient}_ap.tif'), ap_projection.squeeze().numpy().astype(np.float32))
        tiff.imsave(os.path.join(lt_dir, f'{patient}_lt.tif'), lt_projection.squeeze().numpy().astype(np.float32))
        torch.save(ap_projection, os.path.join(ap_dir, f'{patient}_ap.pt'))
        torch.save(lt_projection, os.path.join(lt_dir, f'{patient}_lt.pt'))
        print(f'Processed patient {idx + 1} / {len(os.listdir(root_dir))}: {patient}')