import torch
import torch.nn.functional as F
import torch.optim as optim

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# sigmoid (함수, 직접구현)
w = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(w) + b)))
print(hypothesis, '\n')
hypothesis2 = torch.sigmoid(x_train.matmul(w) + b)
print(hypothesis2, '\n')

# binary crossentropy (함수, 직접구현)
losses = -(y_train * torch.log(hypothesis2) + (1 - y_train) * torch.log(1 - hypothesis2))
print(losses, '\n')
loss = losses.mean()
print(loss, '\n')
loss2 = F.binary_cross_entropy(hypothesis2, y_train)
print(loss2, '\n')

# training
optimizer = optim.SGD([w, b], lr=1)
for epoch in range(1000):
    hypothesis = torch.sigmoid(x_train.matmul(w) + b)
    loss = F.binary_cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('epoch:{} loss:{:.4f}'.format(epoch, loss.item()))
print()

# prediction
hypothesis = torch.sigmoid(x_train.matmul(w) + b)
predition = hypothesis > torch.FloatTensor([0.5])
print(predition, '\n')
print(w, '\n')
print(b, '\n')