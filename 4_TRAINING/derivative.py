import sys, os
import numpy as np



def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h) -f(x-h))/(2*h)


def f(x):
    return (x**2)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return x[0]**2 + x[1]**2

print(numerical_diff(function_2, 5))
