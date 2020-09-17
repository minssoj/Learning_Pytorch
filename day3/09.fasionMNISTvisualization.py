from torchvision import datasets, transforms, utils
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] ='TRUE'


trainset = datasets.FashionMNIST(
    root      = 'MNIST_data/',
    train     = True,
    download  = True,
    transform = transforms.Compose([
                       transforms.ToTensor()
                   ])
)
testset = datasets.FashionMNIST(
    root      = 'MNIST_data/',
    train     = False,
    download  = True,
    transform = transforms.Compose([
                       transforms.ToTensor()
                   ])
)


batch_size = 16

train_loader = data.DataLoader(dataset = trainset, batch_size  = batch_size)
test_loader = data.DataLoader(dataset = testset, batch_size  = batch_size)

dataiter = iter(train_loader)
images, labels = next(dataiter)

img   = utils.make_grid(images, padding=0)
npimg = img.numpy()
plt.figure(figsize=(10, 7))
plt.imshow(np.transpose(npimg, (1,2,0)))
plt.show()

print(labels)

CLASSES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


for label in labels:
    index = label.item()
    print(CLASSES[index])


idx = 1

item_img = images[idx]
item_npimg = item_img.squeeze().numpy()
plt.title(CLASSES[labels[idx].item()])
print(item_npimg.shape)
plt.imshow(item_npimg, cmap='gray')
plt.show()

