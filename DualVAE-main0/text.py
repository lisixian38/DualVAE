import torch
print(f"PyTorch 版本: {torch.__version__}")#1.7.1
print(f"CUDA 是否可用: {torch.cuda.is_available()}")#ture
print(f"CUDA 版本: {torch.version.cuda}")#10.2   CUDA Version: 11.6
print(f"GPU 设备: {torch.cuda.get_device_name(0)}")#NVIDIA A100-PCIE-40GB
