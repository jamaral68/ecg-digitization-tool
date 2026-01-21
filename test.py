import torch
x = torch.rand(5, 3)
print("Is CUDA available?", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current CUDA device:", torch.cuda.current_device())
print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("CUDA device properties:", torch.cuda.get_device_properties(torch.cuda.current_device()))
