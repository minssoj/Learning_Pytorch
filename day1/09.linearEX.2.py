import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

x = init.uniform_(torch.Tensor(1000, 1), -10, 10)
value = init.normal_(torch.Tensor(1000, 1), std=0.2)
y_target = 2 * x +3 + value
model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
cost_func = nn.MSELoss()

for epoch in range(500):
    optimizer.zero_grad()
    hyphothesis = model(x)
    cost = cost_func(hyphothesis, y_target)
    cost.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('epoch:{} cost:{:.4f}'.format(epoch+1, cost.item()))