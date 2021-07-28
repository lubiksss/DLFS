import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x>0, dtype = int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(0,x)

def identity_func(x):
    return x

# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a-c)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a/ sum_exp_a
#     return y

# batch
def softmax(a):
    c = np.max(a, axis = 1).reshape(100,-1)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a, axis = 1).reshape(100,-1)
    y = exp_a/ sum_exp_a
    return y
    ??


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    y2 = sigmoid(x)
    y3 = ReLU(x)
    plt.plot(x,y,x,y2,x,y3)
    plt.show()