import numpy as np
from math import *
from functools import reduce


def fxy(x: float, y: float, c1: float, c2: float) -> float:
    xmy = x - y
    return -(xmy * xmy / c1 + c2)


def reduce_vector(xj: float, vector: np.ndarray, c1: float, c2: float, c3: float) -> float:
    return reduce((lambda acum, y: acum + fxy(xj, y, c1, c2) + c3), vector, 0.0)


def logpx(x: np.ndarray, vector: np.ndarray, c1: float, c2: float, c3: float) -> float:
    sum = reduce((lambda acum, xj: acum + reduce_vector(xj, vector, c1, c2, c3)), vector, 0.0)
    return log(sum)


def kde_intensity(train: np.ndarray, validation: np.ndarray, zigma: float) -> float:
    c1 = 2 * zigma * zigma
    c2 = 0.5 * log(pi * c1)
    c3 = 1.0 / train.shape(0)
