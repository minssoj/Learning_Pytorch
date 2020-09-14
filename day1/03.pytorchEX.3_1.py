import torch

t1 = torch.tensor([1, 2, 3, 4, 5, 6]).view(3, 2)
t2 = torch.tensor([7, 8, 9, 10, 11, 12]).view(2, 3)

t3 = torch.matmul(t1, t2)
print(t3, '\n')

t4 = torch.mm(t1, t2)
print(t4, '\n')

t5 = torch.FloatTensor(2, 4, 3)
t6 = torch.FloatTensor(2, 3, 5)
print(t5.size(), '\n')
print(t6.size(), '\n')

# If input is a (b×n×m)tensor, mat2 is a (b×m×p)tensor, out will be a (b×n×p)tensor.
t7 = torch.bmm(t5, t6)
print(t7.size())