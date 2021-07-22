import matplotlib.pyplot as plt
import sys, os
import numpy as np
from gradient import numerical_gradient
from loss_function import cross_entropy_error
from activation_function import softmax

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self,x):
        return np.dot(x,self.W)

    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)

        return loss



if __name__ == '__main__':
    x = np.array([1,2])
    t = np.array([0,0,1])

    net  = simpleNet()
    print(net.W)
    print(net.predict(x))
    print(net.loss(x,t))
