import torch
import torch.nn as nn
import numpy as np

string = 'hello pytorch. how long can a rnn cell remember? show me your limit!'
chars = 'abcdefghijklmnopqrstuvwxyz ?!.,:;01'
char_list = [i for i in chars]
n_letter = len(char_list)
print('number of chars :', n_letter, '\n')

n_hidden = 35
learning_rate = 0.01
total_epoch = 1000

def stringToOnehot(string):
    start = np.zeros(shape=n_letter, dtype=int)
    end = np.zeros(shape=n_letter, dtype=int)
    start[-2] = 1 #[0,0,.....'0,1,0]
    end[-1] = 1   #[0,0,.....'0,0,1]
    for i in string:
        idx = char_list.index(i)
        odata = np.zeros(shape=n_letter, dtype=int)
        odata[idx] = 1
        start = np.vstack([start, odata])
    output = np.vstack([start, end])
    return output

def onehotToChar(onehot_d):
    onehot = torch.Tensor.numpy(onehot_d)
    return char_list[onehot.argmax()]

'''
data = np.zeros(shape=n_letter, dtype=int)
data[5] = 1
print(onehotToChar(torch.from_numpy(data).int()))
'''

class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.act_fn = nn.Tanh()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.act_fn(self.i2h(combined))
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

rnn = RNNNet(n_letter, n_hidden, n_letter)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr= learning_rate)
one_hot = torch.from_numpy(stringToOnehot(string)).type_as(torch.FloatTensor())
print('one_hot shape:', one_hot.shape, '\n')    # start + string + end

for i in range(total_epoch):
    optimizer.zero_grad()
    hidden = rnn.init_hidden()
    total_loss = 0
    for j in range(one_hot.size()[0]-1):
        input = one_hot[j:j+1, :]
        target = one_hot[j+1]
        output, hidden = rnn(input, hidden)
        loss = loss_func(output.view(-1), target.view(-1))
        total_loss += loss
    total_loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print('epoch:{:04d}/{:04d} total_loss:{:.4f}'.format(i+1, total_epoch, total_loss))