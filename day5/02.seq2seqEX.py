# ==================================================================================
# Machine Translation
# minso.jeong@daum.net
# ==================================================================================
# 참고 : https://gist.github.com/keon/e39d3cbfd80daff498772951fb784f35?fbclid=IwAR3YpPVEq4NdKVcq58a6LCxrZn0FO-Hq1BjlMtoZKLnIjQlhaU19qoKQkyI
import torch
import torch.nn as nn
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'     # matplotlib.pyplot 오류 해결

vocab_size = 256                                # ascii size
x = list(map(ord,'hello'))                      # ascii codes mapping (char -> Dec)
y = list(map(ord, 'hola'))                      # ascii codes mapping (char -> Dec)
print('아스키 코드 x: {}'.format(x))
print('아스키 코드 y: {}\n'.format(y))
x_data = torch.LongTensor(x)
y_data = torch.LongTensor(y)

# 모델 구현
class Seq2SeqNet(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.n_layer = 1
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layer, batch_size, self.hidden_size).zero_()

    def forward(self, inputs, targets):
        initState = self.init_state()
        embedding = self.embedding(inputs).unsqueeze(1)
        encoder_output, encoder_state = self.encoder(embedding, initState)
        decoder_state = encoder_state
        decoder_input = torch.LongTensor([0])  # [start]
        outputs = []
        for i in range(targets.size()[0]):
            decoder_input = self.embedding(decoder_input).unsqueeze(1)
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            foutput = self.fc(decoder_output)
            outputs.append(foutput)
            decoder_input = torch.LongTensor([targets[i]])

        outputs = torch.stack(outputs).squeeze()
        return outputs

seq2seq = Seq2SeqNet(vocab_size, 16)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=1e-3)

loss_data = []
for i in range(1000):
    hypothesis = seq2seq(x_data, y_data)
    loss = loss_func(hypothesis, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_data.append(loss.item())
    if i % 50 == 0:
        print('epoch:{:04d}/{:04d}, loss:{:.4f}'.format(i+1, 1000, loss.item()))
        _, top = hypothesis.data.topk(1,1)
        print([chr(c) for c in top.squeeze().numpy().tolist()])

# 결과 출력        
import matplotlib.pyplot as plt
plt.plot(loss_data)
plt.show()