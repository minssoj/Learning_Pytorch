import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# data 출처 : https://data.kma.go.kr/stcs/grnd/grndTaList.do?pgmNo=70
data = pd.read_csv('../data/weather_data.csv', skiprows=[0, 1, 2, 3, 4, 5], encoding='cp949')
print(data, '\n')

temp = data['평균기온(℃)']
temp.plot()
plt.show()

train_x = temp[:1461]   # 2011년 1월 1일 ~ 2014년 12월 31일
test_x = temp[1461:]    #~2016년 12월 31일

train_x = np.array(train_x)
test_x = np.array(test_x)

ATTR_SIZE = 180
tmp = []
for i in range(0, len(train_x) - ATTR_SIZE):
    tmp.append(train_x[i : i+ATTR_SIZE])
train_x = np.array(tmp)
print(pd.DataFrame(train_x), '\n')

class TSnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(180, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 180)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        y = self.fc4(x)
        return y

model = TSnet()
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    total_loss = 0
    d = []
    for i in range(100):
        index = np.random.randint(0, 1281)
        d.append(train_x[index])
    d = np.array(d, dtype=np.float32)
    d = torch.from_numpy(d)

    optimizer.zero_grad()
    hypothesis = model(d)
    loss = loss_func(hypothesis, d)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

    if epoch % 100 == 0:
        print('epoch:{} total_loss{:.4f}'.format(epoch+1, total_loss))
print('\n')

plt.plot(d.data[0].numpy(), label='origianl')
plt.plot(hypothesis.data[0].numpy(), label='hypothesis')
plt.legend(loc=2)
plt.show()