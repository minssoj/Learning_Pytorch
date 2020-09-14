import torch
import torch.nn as nn


class CustomLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.w = nn.Parameter(torch.FloatTensor(torch.randn(input_size, output_size)), requires_grad=True)
        self.b = nn.Parameter(torch.FloatTensor(torch.randn(output_size)), requires_grad=True)

    def forward(self, x):
        y = torch.mm(x, self.w) + self.b
        return y

x = torch.FloatTensor(torch.randn(16, 10))
CModel = CustomLinear(10, 5)
y = CModel(x)   # forward의 출력 y를 출력
print(y, '\n')

params = [p.size() for p in CModel.parameters()]
print(params, '\n')

params = [p for p in CModel.parameters()]
print(params)