import multiprocessing
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


def sum_probability(train: np.ndarray, validation: np.ndarray, sigma: float) -> float:
    train_shape = train.shape
    k = train_shape[0]
    d = train_shape[1]
    A = log(1.0 / k)
    c1 = 2 * sigma * sigma
    cB = 1 / c1
    C = d * log(pi * c1) / 2.0

    sum = reduce(lambda acum, x: acum + logpx(x, train, A, C, cB), validation, 0.0)
    return sum


def sum_probability_tuple(t: (np.ndarray, np.ndarray, float)) -> float:
    return sum_probability(t[0], t[1], t[2])


def sum_probability_parallel(train: np.ndarray, validation: np.ndarray, sigma: float, cores: int = 8) -> float:
    validation_split = np.array_split(validation, cores)
    data_split = map((lambda x: (train, x, sigma)), validation_split)

    pool = multiprocessing.Pool(cores)
    partial_sums = pool.map(sum_probability_tuple, data_split)
    return sum(partial_sums)


def mean_probability(train: np.ndarray, validation: np.ndarray, sigma: float) -> float:
    m = validation.shape[0]
    sum = sum_probability_parallel(train, validation, sigma)
    return sum / m


if __name__ == '__main__':
    dir = "../data/"

    train_data = LearningData.from_file(cifar, 'train', dir)
    print("Training:", train_data.shape)

    validation_data = LearningData.from_file(cifar, 'validation', dir)
    print("Validation:", validation_data.shape)

    sigma = 0.5

    begin = now()

    print('\nsigma:', sigma, now())

    value = mean_probability(train_data.features, validation_data.features, sigma)
    end = now()
    print('mean log probability:', value)
    print('Time:', end - begin, now())
