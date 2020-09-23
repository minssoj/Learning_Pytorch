# =============================================================================================================
# 2. 아래의 문장의 RNN 모델을 통하여 모델링 한다.
#     sentence = ("if you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.")
#  a. 위의 문장을 이용하여 모델을 학습 시킬 때 10문자씩 잘라서 학습시킨다.
#  b. 학습 완료 후 학습 시 사용했던 입력 데이터를 이용하여 결과물을 출력한다.
# =============================================================================================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:', device)

random.seed(111)
torch.manual_seed(111)
if device == 'cuda:0':
    torch.cuda.manual_seed_all(111)

sentence = "if you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea."
char_list = list(set(sentence))     # 중복 없이 sentence 알파벳 집합 생성
n_letter = len(char_list)
print('number of chars :', n_letter, '\n')
print('고유 문자의 수: {}\n고유 문자 출력 : {}\n'.format(n_letter, char_list))

total_epoch = 5000
chunk_len = 10         # sequence_length
hidden_size = 100
batch_size = 1
num_layer = 1
embedding_size = 70
learning_rate = 0.002

def random_chunk():
    start_index = random.randint(0, len(sentence) - chunk_len)
    end_index = start_index + chunk_len +1
    return sentence[start_index : end_index]
print('chunk의 예시 : \n{}'.format(random_chunk()))
print('='*100 +'\n')

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = char_list.index(string[c])
    return tensor
print('good의 mapping:', char_tensor('good'), '\n')

def random_train_set():
    chunk = random_chunk()
    input = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return input, target

class RNNNet(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.embbeding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.encoder = nn.Embedding(self.input_size, self.embbeding_size)
        self.rnn = nn.GRU(self.embbeding_size, self.hidden_size, self.num_layers)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden):
        x = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(x, hidden)
        y = self.fc(output.view(batch_size, -1))
        return y, hidden

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, batch_size, hidden_size)
        return hidden

model = RNNNet(input_size=n_letter,
               embedding_size=embedding_size,
               hidden_size=hidden_size,
               output_size=n_letter,
               num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

def te_st_fn():
    start_str = 'w'
    input = char_tensor(start_str)
    hidden = model.init_hidden()
    for i in range(chunk_len):
        output, hidden = model(input, hidden)
        output_dist = output.data.view(-1).div(0.8).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = char_list[top_i]
        print(predicted_char, end='')
        input = char_tensor(predicted_char)



for i in range(total_epoch):
    input, label = random_train_set()
    loss = torch.tensor([0]).type(torch.FloatTensor)        # loss 0으로 초기화
    hidden = model.init_hidden()
    optimizer.zero_grad()                                   # gradient descent 직전에 초기화
    for j in range(chunk_len-1):
        x_train = input[j]
        y_train = label[j].unsqueeze(0).type(torch.LongTensor)
        hypothesis, hidden = model(x_train, hidden)
        loss += loss_func(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print('epoch:{:04d}/{:04d}, loss:{:.4f}'.format(i+1, total_epoch, loss.item()/chunk_len))
        te_st_fn()
        print('\n', '-' * 100)

