import numpy as np
from twoLayerNet import *
from dataset.mnist import load_mnist
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image

def softmax(a):
    c = np.max(a, axis = 1).reshape(2,1)
    print(a-c)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a, axis = 1).reshape(2,1)
    y = exp_a/ sum_exp_a
    return y

x = np.array([[1,2,3,4],[8,7,6,5]])
print(softmax(x))

# y = np.array([[4],[8]])

