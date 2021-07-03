import numpy as np
# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1*w1 + x2*w2
#     if tmp > theta:
#         return 1
#     else:
#         return 0

# print(AND(0,0))
# print(AND(0,1))
# print(AND(1,0))
# print(AND(1,1))


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x)+b
    if tmp > 0:
        return 1
    else:
        return 0

# print(AND(0,0))
# print(AND(0,1))
# print(AND(1,0))
# print(AND(1,1))

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3
    tmp = np.sum(w*x)+b
    if tmp > 0:
        return 1
    else:
        return 0

# print(OR(0,0))
# print(OR(0,1))
# print(OR(1,0))
# print(OR(1,1))

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.9
    tmp = np.sum(w*x)+b
    if tmp < 0:
        return 1
    else:
        return 0

# print(NAND(0,0))
# print(NAND(0,1))
# print(NAND(1,0))
# print(NAND(1,1))

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2= OR(x1,x2)
    return AND(s1,s2)

print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))

def XOR2(x1,x2):
    s1 = NAND(NAND(x1,x1),x2)
    s2 = NAND(x1,NAND(x2,x2))
    return NAND(s1,s2)

print(XOR2(0,0))
print(XOR2(0,1))
print(XOR2(1,0))
print(XOR2(1,1))