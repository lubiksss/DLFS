import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# x = np.arange(0,6,0.1)
# y1 = np.sin(x)
# y2 = np.cos(x)

# plt.plot(x,y1,label='sin')
# plt.plot(x,y2,label = 'cos',linestyle = '--')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('sin&cos')
# plt.legend()
# plt.show()

img = imread(base_dir+'../data/lenna.png')
plt.imshow(img)
plt.show()