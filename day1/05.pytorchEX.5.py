import torch
import torch.nn.init as init

t1 = init.uniform_(torch.FloatTensor(3, 4))
print(t1, '\n')
t2 = init.normal_(torch.FloatTensor(3, 4), std=0.2)
print(t2, '\n')
t3 = init.constant_(torch.FloatTensor(3, 4), 100)
print(t3, '\n')
t4 = torch.FloatTensor(torch.randn(3, 4))
print(t4)
