import torch
import numpy as np

# torch -> numpy
t1 = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# t1 = torch.tensor([[1,2], [2,3]], dtype=torch.float32, device='cuda:0')
print(t1)
print(t1.size())
print(t1.numpy(), '\n')

t2 = torch.FloatTensor([4, 6, 7, 8, 9])
print(t2)
print(type(t2.numpy()), '\n')

# numpy -> torch
ndata = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
t3 = torch.from_numpy(ndata)
print(t3)
