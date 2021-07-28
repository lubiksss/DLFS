import matplotlib.pyplot as plt
import sys, os
import numpy as np
from tqdm import tqdm

def function_2(x):
    return x[0]**2 + x[1]**2
    
# def numerical_gradient(f,x):
#     h = 1e-4
#     grad = np.zeros_like(x)
#     print(grad.shape)

#     for idx in range(x.size):
#         tmp_val = x[idx]
#         x[idx] = tmp_val+h
#         fxh1 = f(x)

#         x[idx] = tmp_val-h
#         fxh2 = f(x)

#         grad[idx] = (fxh1 - fxh2)/(2*h)
#         x[idx] = tmp_val
#     return grad

## 이렇게 해야 배치용으로 되는거 아닌가?
def numerical_gradient(f,x):
    h = 1e-4
    tmp = x

    if tmp.ndim == 1:
        pass
    else:
        a,b = x.shape[0],x.shape[1]
        x = x.reshape(-1)

    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val+h
        fxh1 = f(x)

        x[idx] = tmp_val-h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2)/(2*h)
        x[idx] = tmp_val
    
    if tmp.ndim == 1:
        return grad
    else:
        return grad.reshape(a,b)


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad
    return x

if __name__ == '__main__':
    print(numerical_gradient(function_2,np.array([2.0,1.5])))

    # plt.axis((-2,2,-2,2))
    # plt.vlines(np.arange(-2,2.5,0.5), -2,2, linestyles=':')
    # plt.hlines(np.arange(-2,2.5,0.5), -2,2, linestyles=':')

    # for i in np.arange(-2,2.25,0.25):
    #     for j in np.arange(-2,2.25,0.25):
    #         nx = numerical_gradient(function_2,np.array([i,j]))[0]
    #         ny = numerical_gradient(function_2,np.array([i,j]))[1]
    #         plt.annotate('',xy=(i-nx/20,j-ny/20), xytext=(i,j), arrowprops=dict(facecolor = 'black', headwidth = 3,headlength = 3, width = 0.5))

    # plt.show()

    # plt.axis((-4,4,-4,4))
    # for i in range(0,100+1,10):
    #     plt.scatter(*gradient_descent(function_2,np.array([4.0,4.0]),lr=0.01,step_num = i),c='black')
    # plt.show()