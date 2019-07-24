from functools import reduce

from math import *

from learning_data import *


def square_difference(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    r = np.dot(diff, diff)
    return r


def exp_part(x: np.ndarray, y: np.ndarray, A: float, C: float, cB: float) -> float:
    B = cB * square_difference(x, y)
    exponent = A - B - C
    r = exp(exponent)
    return r


def logpx(x: np.ndarray, train: np.ndarray, A: float, C: float, cB: float) -> float:
    sum = reduce((lambda acum, y: acum + exp_part(x, y, A, C, cB)), train, 0.0)
    return log(sum)


def mean_probability(train: np.ndarray, validation: np.ndarray, sigma: float) -> float:
    train_shape = train.shape
    k = train_shape[0]
    m = validation.shape[0]
    d = train_shape[1]
    A = log(1.0 / k)
    c1 = 2 * sigma * sigma
    cB = 1 / c1
    C = d * log(pi * c1) / 2.0

    sum = reduce(lambda acum, x: acum + logpx(x, train, A, C, cB), validation, 0.0)
    return sum / m
