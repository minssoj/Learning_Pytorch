import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 92],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for epoch in range(20):
    for batch_idx, data in enumerate(dataloader):
        batch_x, batch_y = data
        hypothesis = model(batch_x)
        cost = F.mse_loss(hypothesis, batch_y)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print('epoch:{}/{} batch:{}/{} cost:{:.5f}'.format(
            epoch+1, 20, batch_idx+1, len(dataloader), cost.item()))
    print()

