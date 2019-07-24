from functools import reduce

from math import *

from learning_data import *

import multiprocessing


def square_difference(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    r = np.dot(diff, diff)
    return r


def exp_part(x: np.ndarray, y: np.ndarray, c1: float) -> float:
    r = c1 * square_difference(x, y)
    return r


def logpx(x: np.ndarray, train: np.ndarray, c: float, c1: float) -> float:
    sum_exp = reduce((lambda acum, y: acum + exp_part(x, y, c1)), train, 0.0)
    r = log(sum_exp)
    return r


def sum_probability(train: np.ndarray, validation: np.ndarray, sigma: float) -> float:
    train_shape = train.shape
    k = train_shape[0]
    d = train_shape[1]
    c = 1.0 / (k * sigma * sqrt(2 * pi))
    c1 = 1.0 / (2 * sigma * sigma)

    sum = reduce(lambda acum, x: acum + logpx(x, train, c, c1), validation, 0.0)
    return sum


def sum_probability_parallel(train: np.ndarray, validation: np.ndarray, sigma: float, np: int) -> float:
    validation_split = np.array_split(validation, np)

    pool = multiprocessing.Pool(np)
    results = pool.map(sum_probability, )


def mean_probability(train: np.ndarray, validation: np.ndarray, sigma: float) -> float:
    m = validation.shape[0]
    sum = sum_probability(train, validation, sigma)
    return sum / m
