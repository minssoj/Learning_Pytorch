import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 20
learning_rate = 0.0002
num_epoch = 100
img_dir = '../data/cat_and_dog'
img_data = datasets.ImageFolder(img_dir,
                                transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                                ]))
train_loader = data.DataLoader(img_data, batch_size=batch_size, shuffle=True)

def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

class VGG16(nn.Module):
    def __init__(self, base_dim, num_classes=2):
        super().__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),
            conv_2_block(base_dim, 2 * base_dim),
            conv_3_block(2 * base_dim, 4 * base_dim),
            conv_3_block(4 * base_dim, 8 * base_dim),
            conv_3_block(8 * base_dim, 8 * base_dim)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8 * base_dim * 7 * 7, 100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 20),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(20, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        y = self.fc_layer(x)
        return y

model = VGG16(base_dim=16).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
for i in model.named_children():
    print(i)

for i in range(num_epoch):
    for j, (image, label) in enumerate(train_loader):
        x_train = image.to(device)
        y_train = label.to(device)
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()
    print('epoch:{:03d}/{:03d} loss:{:.4f}'.format(i+1,num_epoch, loss.item()))