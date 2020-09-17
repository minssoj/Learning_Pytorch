import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 100
learning_rate = 0.001
num_epoch = 15

mnist_train = datasets.FashionMNIST('../data/MNIST_fasion_data/',
                                    train=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(34),
                                        transforms.CenterCrop(28),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Lambda(lambda x: x.rotate(90)),
                                        transforms.ToTensor()
                                    ]),
                                    target_transform=None,
                                    download=True)
mnist_test = datasets.FashionMNIST('../data/MNIST_fasion_data/',
                                   train=False,
                                   transform=transforms.ToTensor(),
                                   target_transform=None,
                                   download=True)
train_loader = torch.utils.data.DataLoader(dataset = mnist_train, batch_size = batch_size, shuffle = True)

class CNNNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc_layer = nn.Sequential(nn.Linear(7 * 7 * 64, 100), nn.ReLU(), nn.Linear(100, 10))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
                # m.bias.data.fill(0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                # m.bias.data.fill(0)

    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)
        y = self.fc_layer(out)
        return y


model = CNNNet2().to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(num_epoch):
    for j, (image, label) in enumerate(train_loader):
        x_train = image.to(device)
        y_train = label.to(device)

        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()

    if i % 2 == 0:
        print('epoch:{:02d}/{:02d} avg_loss: {:.4f} '.format(i+ 1, num_epoch, loss))