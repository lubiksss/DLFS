import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x)+b
    if tmp > 0:
        return 1
    else:
        return 0

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3
    tmp = np.sum(w*x)+b
    if tmp > 0:
        return 1
    else:
        return 0

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.9
    tmp = np.sum(w*x)+b
    if tmp < 0:
        return 1
    else:
        return 0

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2= OR(x1,x2)
    return AND(s1,s2)