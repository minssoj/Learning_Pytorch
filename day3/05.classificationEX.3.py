import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd

torch.manual_seed(777)
wine = load_wine()
print(wine, '\n')

df = pd.DataFrame(wine.data, columns=wine.feature_names)
print(df.head(10), '\n')

wine_data = wine.data[0:130]
wine_target = wine.target[0:130]
print(wine_target, '\n')

train_x, test_x, train_y, test_y = train_test_split(wine_data, wine_target, test_size=0.2, random_state=48)
train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).long()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).long()
train = TensorDataset(train_x, train_y)
train_loader = DataLoader(train, batch_size=16, shuffle=True)

class CNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 96)
        self.fc2 = nn.Linear(96, 96)
        self.fc3 = nn.Linear(96, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y

model = CNet()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(300):
    total_loss = 0
    for train_x, train_y in train_loader:
        optimizer.zero_grad()
        hypothesis = model(train_x)
        #hypothesis = model.forward(train_x)
        loss = loss_func(hypothesis, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print('epoch:{} total_loss{:.4f}'.format(epoch+1, total_loss))
print()

prediction = torch.max(model(test_x).data, dim=1)[1]
print('prediction:', prediction, '\n')
accuracy = (prediction == test_y).float().mean()
print('accuracy:', accuracy.item())
