import numpy as np
import matplotlib.pyplot as plt


def ReLU(x):
    return np.maximum(0, x)


x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num) * np.sqrt(2)
    a = np.dot(x, w)
    z = ReLU(a)
    activations[i] = z

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + 'layer')
    plt.hist(a.flatten(), 30, range=(0, 1))
    if i != 0:
        ax = plt.gca()
        ax.axes.yaxis.set_visible(False)

plt.show()
