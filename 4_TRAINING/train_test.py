import matplotlib.pyplot as plt
import sys, os
import numpy as np
from twoLayerNet import twoLayerNet
from mnist import load_mnist
from tqdm import tqdm

(x_train, t_train),(x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

train_loss_list = []

iters_num = 10
train_size = x_train.shape[0]
print(train_size)
batch_size = 100
learning_rate = 0.1

network = twoLayerNet(input_size=784, hidden_size = 50, output_size = 10)

for i in tqdm(range(iters_num)):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch,t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

print(train_loss_list)
## 10번으로는 어림없네요. 책에서도 200번은해야 하강하는거보니까 200번해야하는데 너무 오래걸려서 할수가없습니다.
## [6.906919570676965, 6.907429547649088, 6.907291407094997, 6.906877994220504, 6.906630094868526, 6.906329935000064, 6.905610886328409, 6.904962606965217, 6.907608210706769, 6.906339269924711]
