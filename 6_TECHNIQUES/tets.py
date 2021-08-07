from matplotlib import pyplot as plt
import numpy as np


x = np.random.randn(100000)

plt.hist(x)
plt.show()
