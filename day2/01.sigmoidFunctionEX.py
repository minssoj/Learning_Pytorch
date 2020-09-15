import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 01. 기본 sigmoid
x = np.arange(-5., 5., 0.1)
y = sigmoid(x)

plt.figure(0)
plt.plot(x, y, 'g')
plt.plot([0,0],[1.,0.], ':')
plt.title('sigmoid func')
# plt.show()

# 02.sigmoid (ax)
# a가 클수록 step function에 가까워진다.
y1 = sigmoid(0.5 * x)
y2 = sigmoid(x)
y3 = sigmoid(2 * x)
plt.figure(1)
plt.plot(x, y1, 'r')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b')
plt.title('sigmoid func')
# plt.show()

# 03.sigmoid (b)
# b가 클수록 더 위로 올라간다.
y1 = sigmoid(x)
y2 = sigmoid(x + 1)
y3 = sigmoid(x + 2)
plt.figure(2)
plt.plot(x, y1, 'r')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b')
plt.title('sigmoid func')
plt.show()