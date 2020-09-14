import torch
import torch.optim as optim

x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [65]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

w1 = torch.zeros((1, 1), requires_grad=True)
w2 = torch.zeros((1, 1), requires_grad=True)
w3 = torch.zeros((1, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

for epoch in range(1000):
    hypothesis = torch.mm(x1_train, w1) + torch.mm(x2_train, w2) + torch.mm(x3_train, w3) + b
    cost = torch.mean((hypothesis - y_train) ** 2)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('epoch:{} w1:{:.3f} w2:{:.3f} w3:{:.3f} b:{:.3f} cost:{:.4f}'.format(
            epoch+1, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))