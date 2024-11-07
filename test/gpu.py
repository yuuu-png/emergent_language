import torch


print("is available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
