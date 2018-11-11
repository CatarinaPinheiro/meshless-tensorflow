import numpy as np


def exponential (n):
    points = []
    for i in range(n):
        points.append([[i], [np.exp(-i)]])
    return np.array(points)
