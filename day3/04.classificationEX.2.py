import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)

# CASE 1
w = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD([w, b], lr=0.1)

for epoch in range(1000):
    hypothesis = F.softmax(x_train.matmul(w) + b, dim=1)
    loss = (y_one_hot * - torch.log(hypothesis)).sum(dim=1).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('epoch:{} loss:{:.4f}'.format(epoch+1, loss.item()))
print('\n')

# CASE 2
class softmaxClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc1(x)

model = softmaxClass()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    hypothesis = model(x_train)
    loss = loss_func(hypothesis, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('epoch:{} loss:{:.4f}'.format(epoch+1, loss.item()))