#!/usr/bin/env python3

import sys

sys.path.extend(['../', '../kde'])

from kde_big_sigma import sum_probability_parallel
from learning_data import *
from utils import *
import argparse


def apply(sigma: float, train: np.ndarray, test: np.ndarray) -> float:
    m = test.shape[0]

    begin = now()

    print('\nsigma:', sigma, now())

    value = sum_probability_parallel(train, test, sigma) / m

    end = now()
    print('mean log probability:', value)
    print('Time:', end - begin, now())
    return value


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=10000)
    parser.add_argument('-m', type=int, default=10000)
    args = vars(parser.parse_args())

    dir = "../data/"
    data_names = [mnist, cifar]
    sigma = 0.2

    k = args['k']
    m = args['m']

    print('Begin fit', now())
    results = [('data', 'sigma', 'quality')]

    for data_name in data_names:
        print('\nData:', data_name)

        train = LearningData.from_file(data_name, 'train', dir).features[:k]
        print("Training:", train.shape)

        test = LearningData.from_file(data_name, 'test', dir).features[:m]
        print("Test:", test.shape)

        q = apply(sigma, train, test)
        results.append((data_name,sigma,q))

    save_csv(results,'../data/kde_test.csv')
    print('End apply', now())
