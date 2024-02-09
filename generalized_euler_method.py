import matplotlib.pyplot as plt
import numpy as np
import math

def linear (t):
    return t
def quadratic (t):
    return t ** 2
def cosine (t):
    return math.cos(t)


diff_eq = np.array([1, 1, 1, 1])
coefficient_of_t = np.array([[1],
                             [4],
                             [-1],
                             [-1]])


