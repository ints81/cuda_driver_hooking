import torch
import torch.nn as nn


dummy_tensor = torch.randn(64, 3, 32, 32).cuda()
conv = nn.Conv2d(3, 64, kernel_size=3).cuda()

dummy_tensor_relu = torch.randn(64, 3, 32, 32).cuda()
relu = nn.ReLU().cuda()

print("#################### Conv ####################")
dummy_output = conv(dummy_tensor)

print("#################### ReLU ####################")
dummy_output_relu = relu(dummy_tensor_relu)