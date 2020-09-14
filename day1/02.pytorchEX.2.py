import torch

t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([5, 6, 7])
t3 = t1 + 10

print(t3, '\n')
print(t2 ** 2, '\n')

t4 = t1 - t2
print(t4, '\n')

t5 = torch.tensor([[10, 20, 30], [40, 50, 60]])
print(t5, '\n')
print(t5 + t1, '\n')

t6 = torch.linspace(0, 3, 10)   # 0부터 3까지 10개
print(t6, '\n')
print(torch.exp(t6), '\n')
print(torch.log(t6), '\n')

t7 = torch.tensor([[2, 7, 6], [1, 3, 9]])
print(t7, '\n')
print(torch.max(t7), '\n')
print(torch.max(t7, dim=1), '\n')
print(torch.max(t7, dim=1)[0], '\n')
print(torch.max(t7, dim=1)[1], '\n')
