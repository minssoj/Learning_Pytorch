import torch

t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t1, '\n')
print(t1[:, :2], '\n')
print(t1[t1>4], '\n')

t1[:, 2] = 30
print(t1, '\n')

t1[t1>4] = -10
print(t1, '\n')

t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
t3 = torch.tensor([[7, 8, 9], [10, 11, 12]])

t4 = torch.cat([t2, t3], dim=0)
print(t4, '\n')

# 행방향으로 자르기 (ㅣ)
c1, c2, c3, c4 = torch.chunk(t4, 4, dim=0)
print(c1, '\n')
print(c2, '\n')
print(c3, '\n')
print(c4, '\n')

# 열방향으로 자르기 (->)
c1, c2, c3 = torch.chunk(t4, 3, dim=1)
print(c1, '\n')
print(c2, '\n')
print(c3, '\n')