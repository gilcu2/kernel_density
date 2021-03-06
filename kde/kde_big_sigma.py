import multiprocessing
from functools import reduce
from math import *
from kde.utils import *

from kde.learning_data import *


def LSE(x: np.ndarray) -> float:
    max = np.max(x)
    x1 = x - max
    r = max + np.log(np.sum(np.exp(x1)))
    return r


def log_sum(x: np.ndarray, train: np.ndarray, c1: float) -> float:
    diff = train - x
    modules2 = np.square(np.linalg.norm(diff, axis=1))
    by_factor = -modules2 / c1
    r = LSE(by_factor)
    return r


def sum_probability(train: np.ndarray, validation: np.ndarray, sigma: float) -> float:
    train_shape = train.shape
    k = train_shape[0]
    m = validation.shape[0]
    d = train_shape[1]
    c = -log(k) - d * log(sigma) - d / 2 * log(2 * pi)
    c1 = (2 * sigma * sigma)

    sum = reduce(lambda acum, x: acum + log_sum(x, train, c1), validation, 0.0)
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
