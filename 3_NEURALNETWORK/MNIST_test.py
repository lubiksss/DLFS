from PIL import Image
import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train),(x_test, t_test) = load_mnist(flatten= True, normalize = False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

def img_show(img):
    pil_img = Image.fromarray((img))
    pil_img.show()

img = x_train[0]
label = t_train[0]
print(img.shape)
print(label)

img = img.reshape(28,28)
print(img.shape)

img_show(img)