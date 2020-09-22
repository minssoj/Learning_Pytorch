import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)

random.seed(111)
torch.manual_seed(111)
if device == 'cuda:0':
    torch.cuda.manual_seed_all(111)

traing_epochs = 15
batch_size = 100
learning_rate = 0.0002

mnist_train = datasets.MNIST(root='../data/MNIST_data/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)
data_loader = DataLoader(dataset=mnist_train,
                           batch_size=batch_size,
                           shuffle=True)

class AutoEncodoer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)  # 입력의 특징을 3차원으로 압축시킨다
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()  # 픽셀당 0과 1사이의 값을 출력 시킨다
        )

    def forward(self, x):
        eoutput = self.encoder(x)
        y = self.decoder(eoutput)
        return eoutput, y


autoencoder = AutoEncodoer().to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

# noise 생성
def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noise_img = img + noise
    return noise_img

def train(autoencoder, train_loader):
    autoencoder.train()
    avg_loss = 0

    for i, (image, _) in enumerate(train_loader):
        x_data = add_noise(image)  # 이미지에 노이즈를 더한다
        x_data = x_data.view(-1, 784).to(device)
        y = image.view(-1, 784).to(device)

        _, decoded = autoencoder(x_data)
        loss = loss_func(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
    return avg_loss / len(train_loader)

for epoch in range(traing_epochs):
    loss = train(autoencoder, data_loader)
    print('epoch:{:02d}/{:02d}, loss:{:.4f}'.format(epoch+1, traing_epochs, loss))


mnist_test = datasets.MNIST(root='../data/MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

sample_data = mnist_test.data[0].view(-1, 784)
sample_data = sample_data.type(torch.FloatTensor)/255.
original_x = sample_data[0]
noise_x = add_noise(original_x).to(device)
_, recovered_x = autoencoder(noise_x)

original_img = np.reshape(original_x.to(device).data.numpy(), (28, 28))
noise_img = np.reshape(noise_x.to(device).data.numpy(), (28, 28))
recovered_img = np.reshape(recovered_x.to(device).data.numpy(), (28, 28))

# subplot의 다른 방법
f, a = plt.subplots(1, 3, figsize=(15,15))
a[0].set_title('original')
a[0].imshow(original_img, cmap='gray')
a[1].set_title('noise')
a[1].imshow(noise_img, cmap='gray')
a[2].set_title('recovered')
a[2].imshow(recovered_img, cmap='gray')
plt.show()