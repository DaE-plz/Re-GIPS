# import os
# import tqdm
# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader
# from torchvision.utils import save_image
# import projection_dataset
# import u_net
# from torch.utils.data import random_split
# import matplotlib.pyplot as plt
# import tifffile
#
# def save_as_tif(image, output_file):
#     # 确保image是numpy数组
#     image_np = image.cpu().detach().numpy()
#     tifffile.imsave(output_file, image_np)  # 确保参数顺序正确
#     print(f'Image saved as {output_file}')
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# user_dir = '/home/woody/iwi5/iwi5191h'
# project_dir = os.path.join(user_dir, 'dl_gips_forschung')
#
# log_path = os.path.join(project_dir, 'training_log.csv')
# checkpoint_file = 'model_checkpoint.pth'
# checkpoint_path = os.path.join(project_dir, 'checkpoints', checkpoint_file)  # 指向具体文件
# save_path = os.path.join(project_dir, 'train_image')
# #log_path = 'training_log.csv'
# if not os.path.exists(log_path):
#     with open(log_path, 'w') as log_file:
#         log_file.write('Epoch,Train Loss,Validation Loss\n')
#
# #checkpoint_path = 'checkpoints/dl_gips_checkpoint.pth'
# if not os.path.exists('checkpoints'):
#     os.makedirs('checkpoints')
#
# #save_path = 'train_image'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
#
#
#
#
# dataset = projection_dataset.ProjectionTensorDataset(
#     ap_dir=f'/scratch/{os.environ["SLURM_JOB_ID"]}/ap',
#     lt_dir=f'/scratch/{os.environ["SLURM_JOB_ID"]}/lt',
#
# )
# data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
#
# # 计算训练/验证集和测试集的大小
# total_patients = len(dataset)
# test_size = int(total_patients * 0.2)
# train_val_size = total_patients - test_size
#
# # 定义随机种子以确保可复现性 # 保证每次运行代码时，所有的随机操作（比如初始化权重、打乱数据集、随机选择dropout单元等）都会以相同的方式发生
# torch.manual_seed(42)
#
# # 随机划分数据集为训练/验证集和测试集
# train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])
#
#
# # 将20%的训练/验证集数据用作验证
# val_size = int(train_val_size * 0.2)
# train_size = train_val_size - val_size
# train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
#
# # 创建相应的 DataLoader
# train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
#
#
#
# num_classes = 1  # 对于回归问题，类别数设置为1
# net = u_net.UNet(num_classes).to(device)
#
#
# opt = optim.Adam(net.parameters())
# loss_fun = nn.L1Loss()  # 使用L1Loss作为回归任务的损失函数
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path)
#
#     # 加载所有模块的状态字典
#     net.load_state_dict(checkpoint['net_state_dict'])
#     # 加载其他信息，如当前epoch
#     start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始继续训练
#     avg_val_loss = checkpoint['loss']
#
#     print(f"Loaded checkpoint from epoch {start_epoch - 1}")
# else:
#     epoch = 1
#     print("No checkpoint found, starting from scratch.")
#     train_losses = []
#     val_losses = []
#     while epoch < 500:
#         train_loss_total = 0  # 累计整个epoch的损失
#         num_batches = 0  # 记录批次数量
#         for i, (ap_projection, lt_projection) in enumerate(tqdm.tqdm(train_loader)):
#             ap_projection, lt_projection = ap_projection.to(device), lt_projection.to(device)
#             lt_pred = net(ap_projection)
#             train_loss = loss_fun(lt_pred, lt_projection)
#             opt.zero_grad()
#             train_loss.backward()
#             opt.step()
#
#             if i == len(train_loader) - 1:
#                 _ap = ap_projection[0].cpu().detach()
#                 _lt = lt_projection[0].cpu().detach()
#                 _lt_pred = lt_pred[0].cpu().detach()
#
#                 # 使用tifffile保存图像
#                 save_as_tif(_ap, f'{save_path}/{epoch}__ap.tif')
#                 save_as_tif(_lt, f'{save_path}/{epoch}__lt.tif')
#                 save_as_tif(_lt_pred, f'{save_path}/{epoch}__ltpred.tif')
#             # 累计损失和批次数量
#             train_loss_total += train_loss.item()
#             num_batches += 1
#         # 计算并打印平均训练损失
#         avg_train_loss = train_loss_total / num_batches
#
#         # Validation phase after each training epoch
#         net.eval()
#         with torch.no_grad():
#             val_loss_total = 0
#             for ap_projection, lt_projection in val_loader:
#                 ap_projection, lt_projection = ap_projection.to(device), lt_projection.to(device)
#                 lt_pred = net(ap_projection)
#                 val_loss = loss_fun(lt_pred, lt_projection)
#                 val_loss_total += val_loss.item()
#             avg_val_loss = val_loss_total / len(val_loader)
#         with open(log_path, 'a') as log_file:
#             log_file.write(f'{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f}\n')
#
#         print(f'Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')
#
#         # Save checkpoint 每隔20个保存一个checkpoints
#         if epoch % 20 == 0:
#             checkpoint = {
#                 'epoch': epoch,
#                 'net_state_dict': net.state_dict(),
#
#                 'loss': avg_val_loss,
#
#             }
#             torch.save(checkpoint, checkpoint_path)
#             print(f'Checkpoint saved at epoch {epoch}')
#
#         train_losses.append(avg_train_loss)
#         val_losses.append(avg_val_loss)
#
#         epoch += 1
#
#     # Plotting the losses
#     # Plotting the losses
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss Over Epochs')
#     plt.legend()
#     # Save the plot to a file
#     plt.savefig(os.path.join(save_path, 'training_validation_loss_plot.png'))
#     plt.close()  # Close the plot to free up memory
import os
import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import projection_dataset
import u_net
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import tifffile

def save_as_tif(image, output_file):
    image_np = image.cpu().detach().numpy()
    tifffile.imsave(output_file, image_np)
    print(f'Image saved as {output_file}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

user_dir = '/home/woody/iwi5/iwi5191h'
project_dir = os.path.join(user_dir, 'dl_gips_forschung')

log_path = os.path.join(project_dir, 'training_log.csv')
checkpoint_file = 'model_checkpoint.pth'
checkpoint_path = os.path.join(project_dir, 'checkpoints', checkpoint_file)
save_path = os.path.join(project_dir, 'train_image')

if not os.path.exists(log_path):
    with open(log_path, 'w') as log_file:
        log_file.write('Epoch,Train Loss,Validation Loss\n')

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

if not os.path.exists(save_path):
    os.makedirs(save_path)

dataset = projection_dataset.ProjectionTensorDataset(
    ap_dir=f'/scratch/{os.environ["SLURM_JOB_ID"]}/ap',
    lt_dir=f'/scratch/{os.environ["SLURM_JOB_ID"]}/lt',
)
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

total_patients = len(dataset)
test_size = int(total_patients * 0.2)
train_val_size = total_patients - test_size

torch.manual_seed(42)
train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])

val_size = int(train_val_size * 0.2)
train_size = train_val_size - val_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

net = u_net.UNet(num_classes=1).to(device)

opt = optim.Adam(net.parameters())
loss_fun = nn.L1Loss()

min_val_loss = float('inf')

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {start_epoch - 1}")
else:
    start_epoch = 1
    print("No checkpoint found, starting from scratch.")

train_losses = []
val_losses = []
for epoch in range(start_epoch, 500):
    net.train()
    train_loss_total = 0
    num_batches = 0
    for ap_projection, lt_projection in tqdm.tqdm(train_loader):
        ap_projection, lt_projection = ap_projection.to(device), lt_projection.to(device)
        lt_pred = net(ap_projection)
        train_loss = loss_fun(lt_pred, lt_projection)
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        train_loss_total += train_loss.item()
        num_batches += 1

    avg_train_loss = train_loss_total / num_batches

    net.eval()
    val_loss_total = 0
    with torch.no_grad():
        for ap_projection, lt_projection in val_loader:
            ap_projection, lt_projection = ap_projection.to(device), lt_projection.to(device)
            lt_pred = net(ap_projection)
            val_loss = loss_fun(lt_pred, lt_projection)
            val_loss_total += val_loss.item()
    avg_val_loss = val_loss_total / len(val_loader)

    with open(log_path, 'a') as log_file:
        log_file.write(f'{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f}\n')

    print(f'Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')

    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        checkpoint = {
            'epoch': epoch,
            'net_state_dict': net.state_dict(),
            'loss': avg_val_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved at epoch {epoch} with validation loss {avg_val_loss:.4f}')

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.savefig(os.path.join(save_path, 'training_validation_loss_plot.png'))
plt.close()
