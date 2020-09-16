import torch
import torch.nn.functional as F

torch.manual_seed(777)

x = torch.FloatTensor([[1, 2, 3], [4, 1, 2]])
hypothsis = F.softmax(x, dim=1)
print(hypothsis)
print(hypothsis.sum(dim=1), '\n')

x = torch.randn(3, 5, requires_grad=True)
print(torch.log(F.softmax(x, dim=1)))
print(F.log_softmax(x, dim=1), '\n')

hypothesis = F.softmax(x, dim=1)
y = torch.randint(5, (3,)).long()
print(y, '\n')

y_one_hot = torch.zeros_like(hypothesis)
print(y_one_hot, '\n')
print(y.unsqueeze(dim=1), '\n\n')

y_one_hot.scatter_(1, y.unsqueeze(dim=1), 1)
print(y_one_hot, '\n')
print((y_one_hot * -torch.log(F.softmax(x, dim=1))).sum(dim=1), '\n')
print((y_one_hot * -torch.log(F.softmax(x, dim=1))).sum(dim=1).mean(), '\n')
print((y_one_hot * -F.log_softmax(x, dim=1)).sum(dim=1).mean(), '\n')
print(F.cross_entropy(x, y))