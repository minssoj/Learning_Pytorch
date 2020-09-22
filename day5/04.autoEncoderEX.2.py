import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)


traing_epochs = 30
batch_size = 100
learning_rate = 0.0002

mnist_train = datasets.MNIST(root='../data/MNIST_data/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = DataLoader(dataset=mnist_train,
                           batch_size=batch_size,
                           shuffle=True)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2 ,2)                  # 64 * 14 * 14
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2 ,2),                 # 128 * 7 * 7
            nn.Conv2d(128, 256, 3, padding=1),  # 256 * 7 * 7
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        y = x.view(batch_size, -1)
        return y

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),   # 128 * 14 *14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),       # 64 * 14 * 14
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 3, 1, 1),        # 16 * 14 * 14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1),      # 1 * 28 * 28
            nn.ReLU()
        )
    def forward(self, x):
        x = x.view(batch_size, 256, 7, 7)
        x = self.layer1(x)
        y = self.layer2(x)
        return y

encoder = Encoder().to(device)
decoder = Decoder().to(device)
loss_func = nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

for i in range(traing_epochs):
    avg_loss = 0
    total_batch = len(data_loader)
    for j, (x_data_, _) in enumerate(data_loader):
        x_data = x_data_.to(device)
        optimizer.zero_grad()
        eoutput = encoder(x_data)
        hypothesis = decoder(eoutput)
        loss = loss_func(hypothesis, x_data)
        loss.backward()
        optimizer.step()
        avg_loss += loss / total_batch
    print('epoch: {:02d}/{:02d} loss: {:.4f}'.format(i + 1, traing_epochs, avg_loss))

out_img = torch.squeeze(hypothesis.data)
for i in range(5):
    plt.subplot(1, 2, 1)
    plt.imshow(torch.squeeze(x_data[i]).numpy(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(out_img[i].numpy(), cmap='gray')
    plt.show()