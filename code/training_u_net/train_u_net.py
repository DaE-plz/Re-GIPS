import os
import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import code.dataset_prepare.projection_dataset
import code.module_u_net.u_net


import tifffile

def save_as_tif(image, output_file):
    # 确保image是numpy数组
    image_np = image.cpu().detach().numpy()
    tifffile.imsave(output_file, image_np)  # 确保参数顺序正确
    print(f'Image saved as {output_file}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
save_path = 'train_image'
if not os.path.exists(save_path):
    os.makedirs(save_path)


dataset = code.dataset_prepare.projection_dataset  .ProjectionTensorDataset(ap_dir='G:/dataset/projection_ap', lt_dir='G:/dataset/projection_lt')
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

num_classes = 1  # 对于回归问题，类别数设置为1
net = code.module_u_net.u_net.UNet(num_classes).to(device)

if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print('successful load weight！')
else:
    print('not successful load weight')

opt = optim.Adam(net.parameters())
loss_fun = nn.L1Loss()  # 使用L1Loss作为回归任务的损失函数

epoch = 1
while epoch < 2:
    for i, (ap_projection, lt_projection) in enumerate(tqdm.tqdm(data_loader)):
        ap_projection, lt_projection = ap_projection.to(device), lt_projection.to(device)
        ap_pred = net(ap_projection)
        train_loss = loss_fun(ap_pred, lt_projection)
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        if i % 1 == 0:
            print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

        if i % 10 == 0:
            _ap = ap_projection[0].cpu().detach()
            _lt = lt_projection[0].cpu().detach()
            _lt_pred = ap_pred[0].cpu().detach()

            # 使用tifffile保存图像
            save_as_tif(_ap, f'{save_path}/{epoch}_{i}_ap.tif')
            save_as_tif(_lt, f'{save_path}/{epoch}_{i}_lt.tif')
            save_as_tif(_lt_pred, f'{save_path}/{epoch}_{i}_ltpred.tif')
    if epoch % 20 == 0:
        torch.save(net.state_dict(), weight_path)
        print('save successfully!')
    epoch += 1
