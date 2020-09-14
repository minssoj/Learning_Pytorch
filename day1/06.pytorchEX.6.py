import torch

t3 = torch.zeros(2, 3)
print(t3.size(), '\n')
print(torch.unsqueeze(t3, dim=0).size(), '\n')
print(torch.unsqueeze(t3, dim=1).size(), '\n')
print(torch.unsqueeze(t3, dim=2).size(), '\n')