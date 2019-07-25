import multiprocessing
from functools import reduce

from math import *

from learning_data import *


def square_difference(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    r = np.dot(diff, diff)
    return r


def exp_part(x: np.ndarray, y: np.ndarray, c1: float) -> float:
    frac = np.float128(-square_difference(x, y) / c1)
    r = np.exp(frac)

    return r


def logpx(x: np.ndarray, train: np.ndarray, c1: float) -> float:
    sum_exp = reduce((lambda acum, y: acum + exp_part(x, y, c1)), train, 0.0)
    r = np.log(sum_exp)
    return r


def sum_probability(train: np.ndarray, validation: np.ndarray, sigma: float) -> float:
    train_shape = train.shape
    k = train_shape[0]
    m = validation.shape[0]
    d = train_shape[1]
    c = -log(k) - d * log(sigma) - d / 2 * log(2 * pi)
    c1 = (2 * sigma * sigma)

    sum = reduce(lambda acum, x: acum + logpx(x, train, c1), validation, 0.0)
    return m * c + sum


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

    print("My formula")

    train_data = LearningData.from_file(cifar, 'train', dir)

    validation_data = LearningData.from_file(cifar, 'validation', dir)

    sigmas = [0.08, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

    k = 10000
    m = 100

    train = train_data.features[:k]
    validation = validation_data.features[:m]

    print("Training:", train.shape)
    print("Validation:", validation.shape)

    begin = now()

    for sigma in sigmas:
        print('\nsigma:', sigma, now())

        value = sum_probability(train, validation, sigma)
        end = now()
        print('sum log probability:', value)
        print('Time:', end - begin, now())
