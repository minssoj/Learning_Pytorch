# ==================================================================================
# Text generation with an RNN
# minso.jeong@daum.net
# ==================================================================================
# 참고 : RNN을 활용한 문자열 생성
# https://www.tensorflow.org/tutorials/text/text_generation?hl=ko
# 문자 시퀀스가 주어지면 시퀀스의 다음 문자를 예측하는 모델

import torch
import torch.nn as nn
import unidecode
import string
import random

file = unidecode.unidecode(open('../data/coriolanus.txt').read())
file_len = len(file)
print('txt에 있는 문자의 수 : {}\n'.format(file_len))

total_epoch = 2000
chunk_len = 200         # sequence_length
hidden_size = 100
batch_size = 1
num_layer = 1
embedding_size = 70
learning_rate = 0.002

all_charaters = string.printable
n_characters = len(all_charaters)
print('고유 문자의 수: {}\n고유 문자 출력 : {}\n'.format(n_characters, all_charaters))

def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len +1
    return file[start_index : end_index]
print('chunk의 예시 : \n{}'.format(random_chunk()))
print('='*100 +'\n')

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_charaters.index(string[c])
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

model = RNNNet(input_size=n_characters,
               embedding_size=embedding_size,
               hidden_size=hidden_size,
               output_size=n_characters,
               num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

def te_st_fn():
    start_str = 'b'
    input = char_tensor(start_str)
    hidden = model.init_hidden()
    for i in range(200):
        output, hidden = model(input, hidden)
        output_dist = output.data.view(-1).div(0.8).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = all_charaters[top_i]
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
        print('\n', '=' * 100)