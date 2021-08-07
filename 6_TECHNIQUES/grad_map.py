import numpy as np
import matplotlib.pyplot as plt


def function(x, y):
    # return 1/20*(x**2) + (y**2)
    return (x**2) + (y**2)


def func_grad(x, y):
    # return (1/10*x) + 2*y
    return 2*x + 2*y


def grad_map(function, func_grad, rate):
    x = np.arange(0, 11, 0.1)
    arrowprops = dict(color='black',
                      headwidth=2,
                      headlength=4,
                      width=0.25)
    dist = 0.05
    plt.axis([-10, 10, -10, 10])
    for y in range(-10, 11, 1):
        for x in range(-10, 11, 1):
            plt.annotate('', xy=(x-dist*func_grad(x, 0), y-dist*func_grad(0, y)),
                         xytext=(x, y), arrowprops=arrowprops)

    x = []
    y = []
    rate = rate
    start = np.array([-7.0, 2.0])
    for i in range(100):
        x.append(start[0])
        y.append(start[1])
        start = [start[0] - rate *
                 func_grad(start[0], 0), start[1] - rate*func_grad(0, start[1])]
    plt.plot(x, y, 'red')
    plt.scatter(x, y)
    plt.show()


grad_map(function, func_grad, 0.1)
