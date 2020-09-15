import numpy as np

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tp = np.sum(w * x) + b
    if tp <= 0:
        return 0
    else:
        return 1

input_data = [(0, 0), (0, 1), (1, 0), (1, 1)]

for x in input_data:
    y = OR(x[0], x[1])
    print('{} => {}'.format(x, y))

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tp = np.sum(w * x) + b
    if tp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tp = np.sum(w * x) + b
    if tp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    z1 = OR(x1, x2)
    z2 = NAND(x1, x2)
    y = AND(z1, z2)
    return y
