import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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

mnist_train = dset.MNIST(root='../data/MNIST_data/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)
mnist_test = dset.MNIST(root='../data/MNIST_data/',
                        train=False,
                        transform=transforms.ToTensor(),
                        download=True)
data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(traing_epochs):
    avg_loss = 0
    total_batch = len(data_loader)

    for x_train, y_train in data_loader:
        x_train = x_train.view(-1, 28 * 28).to(device)
        y_train = y_train.to(device)

        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()

        avg_loss += loss / total_batch
    print('epoch:{} avg_loss:{:.4f}'.format(epoch+1, avg_loss))
print()

with torch.no_grad():
    x_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    y_test = mnist_test.test_labels.to(device)
    prediction = model(x_test)
    correction_prediction = torch.argmax(prediction, dim=1) == y_test
    accuracy = correction_prediction.float().mean()
    print('accuracy:', accuracy.item(), '\n')

    r = random.randint(0, len(mnist_test) - 1)
    x_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('label:', y_single_data.item())
    s_prediction = model(x_single_data)
    print('prediction:', torch.argmax(s_prediction, dim=1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()