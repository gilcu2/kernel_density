#!/usr/bin/env python3

import sys

sys.path.extend(['../', '../kde'])

from kde_big_sigma import sum_probability_parallel
from learning_data import *
from utils import *
import argparse


def apply(sigma: float, train: np.ndarray, test: np.ndarray, cores_ini: int = 1, cores_fin=9) -> float:
    m = test.shape[0]

    results = [('cores', 'time')]
    for cores in range(cores_ini, cores_fin):
        begin = now()

        print('\nsigma:', sigma, 'cores:', cores, now())

        value = sum_probability_parallel(train, test, sigma, cores) / m

        time = now() - begin

        print('mean log probability:', value)
        print('Time:', time, time.seconds, now())
        results.append((cores, time.seconds))
    return value, results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=10000, help='training size')
    parser.add_argument('-m', type=int, default=10000, help='testing size')
    parser.add_argument('-b', type=int, default=8, help='Initial number of processors')
    parser.add_argument('-e', type=int, default=9, help='End number of processors')
    args = vars(parser.parse_args())

    dir = "../data/"
    data_names = [mnist, cifar]
    sigma = 0.2

    k = args['k']
    m = args['m']
    ini_cores = args['b']
    end_cores = args['e']

    print('Begin fit', now())
    results = [('data', 'sigma', 'quality')]

    for data_name in data_names:
        print('\nData:', data_name)

        train = LearningData.from_file(data_name, 'train', dir).features[:k]
        print("Training:", train.shape)

        test = LearningData.from_file(data_name, 'test', dir).features[:m]
        print("Test:", test.shape)

        q, times = apply(sigma, train, test, ini_cores, end_cores)
        results.append((data_name, sigma, q))
        save_csv(times, '../data/' + data_name + '-time.csv')

    save_csv(results, '../data/kde_test.csv')
    print('End apply', now())
