import os

# 指定源目录路径
source_directory_path = r'G:\dataset\LIDC-IDRI'

# 获取所有子文件夹并按字母顺序排序
subfolders = [f.path for f in os.scandir(source_directory_path) if f.is_dir()]
subfolders.sort()

# 检查是否有足够的子文件夹以匹配到LIDC-IDRI-0819
if len(subfolders) < 819:
    print(f"只找到 {len(subfolders)} 个子文件夹，不足以命名到 LIDC-IDRI-0819。")
else:
    # 对每个子文件夹重命名
    for i, folder in enumerate(subfolders, 1):
        new_name = f"LIDC-IDRI-{i:04d}"  # 生成新的文件夹名（如 LIDC-IDRI-0001）
        new_path = os.path.join(source_directory_path, new_name)
        os.rename(folder, new_path)
        print(f"已将 {folder} 重命名为 {new_path}")

print("重命名完成。")
