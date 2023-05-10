import torch
import torch.nn as nn

print("#################### Linear ####################")
dummy_tensor_linear = torch.randn(10, 10).cuda()
linear = nn.Linear(10, 10, bias=True).cuda()

dummy_output_linear = linear(dummy_tensor_linear)

print("#################### Conv ####################")
dummy_tensor = torch.randn(64, 3, 32, 32).cuda()
conv = nn.Conv2d(3, 64, kernel_size=3).cuda()

dummy_output = conv(dummy_tensor)

print("#################### BatchNorm ####################")
dummy_tensor_bn = torch.randn(64, 3, 32, 32).cuda()
bn = nn.BatchNorm2d(3).cuda()

dummy_output_bn = bn(dummy_tensor_bn)

print("#################### ReLU ####################")
dummy_tensor_relu = torch.randn(64, 3, 32, 32).cuda()
relu = nn.ReLU().cuda()

dummy_output_relu = relu(dummy_tensor_relu)

print("#################### Matmul ####################")
dummy_tensor1 = torch.randn(10, 10).cuda()
dummy_tensor2 = torch.randn(10, 10).cuda()

dummy_output_matmul = torch.matmul(dummy_tensor1, dummy_tensor2)


print("#################### Elementwise Addtion ####################")
dummy_tensor1 = torch.randn(10, 10).cuda()
dummy_tensor2 = torch.randn(10, 10).cuda()

dummy_output_matmul = dummy_tensor1 + dummy_tensor2