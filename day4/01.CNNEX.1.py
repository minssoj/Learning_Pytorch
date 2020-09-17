import torch
import torch.nn as nn

inputs = torch.Tensor(1, 1, 28, 28)

# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
conv1 = nn.Conv2d(1, 32, 3, padding=1)
print(conv1)
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
print(conv2)
pool = nn.MaxPool2d(kernel_size=2)
print(pool, '\n')

output = conv1(inputs)
print(output.size(), '\n')
output = conv2(output)
print(output.size(), '\n')
output = pool(output)
print(output.size(), '\n')
output = output.view(output.size(0), -1)
print(output.size(), '\n')

