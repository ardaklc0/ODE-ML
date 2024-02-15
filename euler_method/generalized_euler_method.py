import matplotlib.pyplot as plt
import numpy as np
import math
from math import cos
from math import sin
import pandas as pd

def linear (t):
    return t
def quadratic (t):
    return t ** 2
def cosine (t):
    return math.cos(t)

"""
# step_size = 0.001
# epoch = 100
# t_0 = 0
# diff_eq = lambda t: np.array([1, 4*t, -t**2, -cos(t)])
# f = lambda t: np.array([sin(t), cos(t), t**2, -4*t])
# y_0 = np.array([2, -1, 3])

# f_0 = np.dot(f(t_0), np.concatenate(([1], y_0)))
# print("f_0:", f_0)
# recurrence_relation = step_size * np.concatenate((y_0[1:], [f_0]))
# print("recurrence_relation:", recurrence_relation)
# y_1 = y_0 + recurrence_relation
# print("y_1:", y_1)

# t_1 = t_0 + step_size
# f_1 = np.dot(f(t_1), np.concatenate(([1], y_1)))
# print("f_1:", f_1)
# recurrence_relation = step_size * np.concatenate((y_1[1:], [f_1])) 
# print("recurrence_relation:", recurrence_relation)
# y_2 = y_1 + recurrence_relation
# print("y_2:", y_2)

# t_2 = t_1 + step_size
# f_2 = np.dot(f(t_1), np.concatenate(([1], y_2)))
# print("f_2:", f_2)
# recurrence_relation = step_size * np.concatenate((y_2[1:], [f_2])) 
# print("recurrence_relation:", recurrence_relation)
# y_3 = y_2 + recurrence_relation
# print("y_3:", y_3)

# step_size = 0.001
# epoch = 100
# t = 0
# df = pd.DataFrame() 
# diff_eq = lambda t: np.array([1, 4*t, -t**2, -cos(t)])
# f = lambda t: np.array([sin(t), cos(t), t**2, -4*t])
# y = np.array([2, -1, 3])

# for i in range(epoch):
#     t = t + i * step_size
#     f_t = np.dot(f(t), np.concatenate(([1], y)))
#     recurrence_relation = step_size * np.concatenate((y[1:], [f_t]))
#     y = y + recurrence_relation
#     #print("y:", y)
#     df = df._append({'y'  : y[0], 
#                      "y'" : y[1], 
#                      "y''": y[2]}, ignore_index=True)

# pd.options.display.max_columns = None
# pd.options.display.max_rows = None
# print(df)
"""

def euler_method(t, step_size, epoch, f, initial_conditions):
    y = initial_conditions
    df = pd.DataFrame()
    if len(y) == 1:
        for i in range(epoch):
            y = y + step_size * y
            df = df._append({'y'  : y[0]}, ignore_index=True)
        return df
    else:
        for i in range(epoch):
            t = t + i * step_size
            f_t = np.dot(f(t), np.concatenate(([1], y)))
            recurrence_relation = step_size * np.concatenate((y[1:], [f_t]))
            y = y + recurrence_relation
            df = df._append({'y'  : y[0], 
                            "y'" : y[1], 
                            "y''": y[2]}, ignore_index=True)
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None
        return df

df = euler_method(0, 0.001, 10000, lambda t: np.array([sin(t), cos(t), t**2, -4*t]), np.array([2, -1, 3]))
print(df)
# df = euler_method(0, 0.001, 10000, lambda t: np.array([0]), np.array([1]))
# print(df)