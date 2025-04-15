import torch
import torchvision

# Print the versions
print(f'PyTorch version: {torch.__version__}')
print(f'torchvision version: {torchvision.__version__}')

# Create a tensor and check if CUDA is available
x = torch.rand(5, 3)
print(f'Tensor: {x}')

# Check CUDA availability
if torch.cuda.is_available():
    print('CUDA is available! Running on GPU!')
    x = x.cuda()
    print(f'Tensor on GPU: {x}')
else:
    print('CUDA is not available. Running on CPU.')