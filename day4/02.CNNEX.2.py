import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)

random.seed(111)
torch.manual_seed(111)
if device == 'cuda:0':
    torch.cuda.manual_seed_all(111)

training_epochs = 15
batch_size = 100
mnist_train = datasets.MNIST(root='../data/MNIST_data/',
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True)
mnist_test = datasets.MNIST(root='../data/MNIST_data/',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7 * 7 * 64, 10, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        y = self.fc(out)
        return y

model = CNNNet().to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(training_epochs):
    avg_loss = 0
    total_batch = len(data_loader)
    # training
    for x_train, y_train in data_loader:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()
        avg_loss += loss/ total_batch

    # evaluation
    with torch.no_grad():
        x_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
        y_test = mnist_test.test_labels.to(device)

        prediction = model(x_test)
        correct_prediction = torch.argmax(prediction, dim=1) == y_test
        accuracy = correct_prediction.float().mean()
    print('epoch:{:02d}/{:02d} avg_loss: {:.4f} accuracy:{:.3f}'.format(epoch + 1, training_epochs, avg_loss, accuracy.item()))