import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

traing_epochs = 20
batch_size = 100
learning_rate = 0.0002

mnist_train = datasets.MNIST(root='../data/MNIST_data/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = DataLoader(dataset=mnist_train,
                           batch_size=batch_size,
                           shuffle=True)

class AutoEncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 20)
        self.decoder = nn.Linear(20, 784)

    def forward(self, x):
        x = x.view(batch_size, -1)
        eoutput = self.encoder(x)
        y = self.decoder(eoutput).view(batch_size, 1, 28, 28)
        return y

AEM = AutoEncoderNet()
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(AEM.parameters(), lr=learning_rate)

for i in range(traing_epochs):
    avg_loss = 0
    total_batch = len(data_loader)
    for j, (x_data, y_data) in enumerate(data_loader):
        optimizer.zero_grad()
        hypothesis = AEM(x_data)
        loss = loss_func(hypothesis, x_data)
        loss.backward()
        optimizer.step()
        avg_loss += loss/total_batch
    print('epoch: {:02d}/{:02d} loss: {:.4f}'.format(i+1, traing_epochs, loss.item()))

out_img = torch.squeeze(hypothesis.data)
for i in range(10):
    plt.subplot(1, 2, 1)
    plt.imshow(torch.squeeze(x_data[i]).numpy(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(out_img[i].numpy(), cmap='gray')
    plt.show()
