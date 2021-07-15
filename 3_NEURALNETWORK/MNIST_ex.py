from PIL import Image
import numpy as np
import sys,os
import pickle
import time
sys.path.append(os.pardir)
from lib.activation_function import *
from dataset.mnist import load_mnist

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=True)
    return x_test, t_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3)+b3
    y = softmax(a3)

    return y

data_list, label_list = get_data()

network = init_network()
cnt = 0
batch_size = 100

starttime = time.time()

for x in range(len(data_list)):
    predictnum = np.argmax(predict(network, data_list[x]))
    label = label_list[x]
    
    if predictnum == label:
        cnt +=1

print(cnt/len(data_list))
print(f'one : {time.time()-starttime}sec')

cnt = 0

starttime = time.time()

for x in range(0,len(data_list),batch_size):
    predictnum = predict(network, data_list[x:x+batch_size])
    # print(predictnum)
    predictnum = np.argmax(predictnum, axis = 1)
    # print(predictnum)
    label = label_list[x:x+batch_size]

    cnt+= np.sum(predictnum == label)

print(cnt/len(data_list))
print(f'batch : {time.time()-starttime}sec')


    