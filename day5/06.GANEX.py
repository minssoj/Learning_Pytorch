import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

epochs = 50
batch_size = 100

trainset = datasets.FashionMNIST(
    root      = '../data/MNIST_fasion_data/',
    train     = True,
    download  = True,
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
)
train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True)

# Generative
G = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Tanh()
).to(device)

# discriminative
D = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()
).to(device)

loss_func = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

for epoch in range(epochs):
    for i, (image_, _) in enumerate(train_loader):
        image = image_.to(device)
        image = image.reshape(batch_size, -1)
        real_label = torch.ones(batch_size, 1).to(device)           # 주의 to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)          # 주의 to(device)
        
        # Discriminative loss 계산
        outputs = D(image)
        d_loss_real = loss_func(outputs, real_label)
        z = torch.randn(batch_size, 64).to(device)                  # 주의 to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = loss_func(outputs, fake_label)
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = loss_func(outputs, real_label)
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print('epoch:{:02d}/{:02d} d_loss:{:.4f}  g_loss:{:.4f}'.format(epoch+1, epochs, d_loss.item(), g_loss.item()))

# 학습 결과 test
z = torch.randn(batch_size, 64).to(device)                          # 주의 to(device)
fake_images = G(z)

# 결과 출력
import numpy as np
import matplotlib.pyplot as plt
for i in range(10):
    fake_images_img = np.reshape(fake_images.to('cpu').data.numpy()[i], (28, 28))
    plt.imshow(fake_images_img, cmap='gray')
    plt.show()