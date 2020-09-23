# =============================================================================================================
# 1.주어지 ImageData를 이용하여 CNN 모델을 구성한다.(제공된 이미지는 벌과 개미 이미지임)
#    a. 학습용 데이터와 테스트용 데이터를 구분한다.
#    b. 학습이 완료 된 후 테스트 데이터를 이용하여 모델의 정확도를 출력한다.
# =============================================================================================================
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

random.seed(111)
torch.manual_seed(111)
if device == 'cuda:0':
    torch.cuda.manual_seed_all(111)

batch_size = 20
learning_rate = 0.00001
num_epoch = 100

img_dir = '../data/hymenoptera_data'

# Training & Test data (label 0:ants,1:bees)
img_data = datasets.ImageFolder(img_dir,
                                transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                                ]))

# ============================ 학습 데이터와 테스트 데이터 분리 코드 ============================
train_data, test_data = data.random_split(img_data, [360, 37])
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(train_data, batch_size=1, shuffle=True)
# ============================ 학습 데이터와 테스트 데이터 분리 코드 ============================

# model
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

# training
for i in range(num_epoch):
    avg_loss = 0
    total_batch = len(train_loader)
    for j, [image, label] in enumerate(train_loader):
        x_train = image.to(device)
        y_train = label.to(device)
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()
        avg_loss += loss/total_batch
    print('epoch:{:03d}/{:03d} loss:{:.4f}'.format(i+1, num_epoch, avg_loss))

# test
with torch.no_grad():
    correct = 0
    for image, label in test_loader:
        image = image.cuda()
        label = label.cuda()
        hypothesis = model(image)
        if torch.argmax(hypothesis, dim=1) == label:
            correct += 1
    accuracy = correct/len(test_loader)
print('accuracy: {:.4f} '.format(accuracy))
