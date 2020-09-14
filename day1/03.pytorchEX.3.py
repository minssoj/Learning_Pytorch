import torch

t1 = torch.tensor([1, 2, 3, 4, 5, 6])
print(t1, '\n')

t2 = t1.view(2, 3)              # 2 by 3으로...
print(t2, '\n')
print(t1.reshape(2, 3), '\n')

t3 = torch.tensor(([[1, 2], [3, 4], [5, 6]]))
print(t3, '\n')
print(t3.size(), '\n')
print(t3.view(-1), '\n')                        # 6
print(t3.view(1, -1), '\n')                     # 1 by 6

t4 = torch.tensor(([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
print(t4, '\n')
print(t4.size(), '\n')
print(t4.view(-1), '\n')
print(t4.view(1, -1), '\n')
print(t3.view(2, -1), '\n')

t5 = torch.tensor([[1, 2, 3], [3, 4, 6]])
t6 = torch.tensor([[10, 20, 30], [40, 50, 60]])
print(t5, '\n')
print(t6, '\n')

print(torch.cat([t5, t6], dim=1), '\n')
print(torch.cat([t5, t6], dim=0), '\n')