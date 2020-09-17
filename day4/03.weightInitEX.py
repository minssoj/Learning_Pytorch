import numpy as np
import matplotlib.pyplot as plt

# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLu(x):
    return np.maximum(0, x)

input_data = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}
x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    # 참고 : https://gomguard.tistory.com/184
    # w = np.random.randn(node_num, node_num)
    # 양극단으로 쏠리는 문제 solution 1 : std값 이용
    # w = np.random.randn(node_num, node_num) *0.01
    # 양극단으로 쏠리는 문제 solution 2 : xavier 방식 (ReLU외의 경우 사용)
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # 양극단으로 쏠리는 문제 solution 3 : kaming he 방식 (ReLU의 경우 a = 0)
    a = 0
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / ((1 + a * a) * node_num))

    a = np.dot(x, w)
    output = sigmoid(a)
    activations[i] = output

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + '-layer')
    if i !=0 :
        plt.yticks([],[])
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()
