# ==========================================================================================
# 아래 코드를 이용하여 fashionMNIST 데이터를 다운로드 받아 모델을 구성한다
# trainset = datasets.FashionMNIST(
#     root      = 'MNIST_data/',
#     train     = True,
#     download  = True,
#     transform = transforms.Compose([transforms.ToTensor()])
# )
# testset = datasets.FashionMNIST(
#     root      = 'MNIST_data/',
#     train     = False,
#     download  = True,
#     transform = transforms.Compose([transforms.ToTensor()])
# )
# -이미지의 종류는 총 10가지로 아래와 같다.
# 0:'T-shirt/top',1: 'Trouser',2: 'Pullover',3: 'Dress',4: 'Coat',
# 5: 'Sandal',    6: 'Shirt',  7: 'Sneaker', 8: 'Bag',  9: 'Ankle boot
# ==========================================================================================
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import os
import torch.nn as nn
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:', device)
print('cuda index:',torch.cuda.current_device())
print('gpu 개수:',torch.cuda.device_count())
print('graphic name:', torch.cuda.get_device_name())
cuda = torch.device('cuda')
print(cuda)

random.seed(111)
torch.manual_seed(111)
if device == 'cuda:0':
    torch.cuda.manual_seed_all(111)
traing_epochs = 30
batch_size = 100

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data/MNIST_fasion_data/',
               train=True,
               download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.5,), (0.5,))
               ])),
    batch_size=batch_size, shuffle=True
)
mnist_test = datasets.FashionMNIST('../data/MNIST_fasion_data/',
               train=False,
               download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.5,), (0.5,))
               ]))

test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=True)

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
        x = F.dropout(x, training=self.training, p=self.dropout_p)
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
print()


CLASSES = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
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

    print('label:', CLASSES[y_single_data.item()])
    s_prediction = model(x_single_data)
    print('prediction:', CLASSES[torch.argmax(s_prediction, dim=1).item()])

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()