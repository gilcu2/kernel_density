from functools import reduce

import numpy as np
from math import *


def square_difference(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    return np.dot(diff, diff)


def logpx(x: np.ndarray, train: np.ndarray, A: float, C: float, cB: float) -> float:
    sum = reduce((lambda acum, y: acum + exp(A + C + cB * square_difference(x, y))), train, 0.0)
    return log(sum)


def mean_probability(train: np.ndarray, validation: np.ndarray, sigma: float) -> float:
    train_shape = train.shape
    k = train_shape[0]
    m = validation.shape[0]
    d = train_shape[1]
    A = log(1.0 / k)
    c1 = 2 * sigma * sigma
    cB = 1 / c1
    C = -d / 2.0 * log(pi * c1)

    sum = reduce(lambda acum, x: acum + logpx(x, train, A, C, cB), validation, 0.0)
    return sum / m
