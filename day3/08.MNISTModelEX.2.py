import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:', device)
print('cuda index:',torch.cuda.current_device())
print('gpu 개수:',torch.cuda.device_count())
print('graphic name:', torch.cuda.get_device_name())
cuda = torch.device('cuda')
print(cuda, '\n')

random.seed(111)
torch.manual_seed(111)
if device == 'cuda:0':
    torch.cuda.manual_seed_all(111)

traing_epochs = 15
batch_size = 100

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/MNIST_data/',
               train=True,
               download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))   # first : mean, Second : std
               ])),
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/MNIST_data/',
               train=False,
               download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])),
    batch_size=batch_size, shuffle=True
)

import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, dropout_p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout_p = dropout_p

    def forward(self,x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout_p)  # training시에만 적용
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training, p=self.dropout_p)
        y = self.fc3(x)
        return y

model = Net(dropout_p=0.2).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        hypothesis = model(data)
        loss = F.cross_entropy(hypothesis, target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            hypothesis = model(data)
            test_loss += F.cross_entropy(hypothesis, target).item()
            pred = hypothesis.max(1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100 * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for epoch in range(1, traing_epochs+1):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print('epoch:{} loss:{:.4f} accuracy:{:.3f}%'.format(epoch, test_loss, test_accuracy))

