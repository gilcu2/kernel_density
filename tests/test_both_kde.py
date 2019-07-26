import numpy as np

from kde import kde_big_sigma
from kde import kde_small_sigma
from kde.input_data import cifar
from kde.learning_data import LearningData


def test_sum_probability_sample():
    train = np.array([
        [0.1, 0.1],
        [0.4, 0.1],
    ])
    validation = np.array([
        [0.2, 0.1],
        # [0.2, 0.7],
    ])

    sigma = 1.0

    r_small_sigma = kde_small_sigma.sum_probability(train, validation, sigma)
    r_big_sigma = kde_big_sigma.sum_probability(train, validation, sigma)
    print('Results', r_small_sigma, r_big_sigma)
    assert abs(r_small_sigma - r_big_sigma) < 0.02


def test_sum_probability_small_data():
    train = np.array([
        [0.1, 0.1],
        [0.2, 0.2],
    ])
    validation = np.array([
        [0.1, 0.1],
        [0.2, 0.2],
    ])

    sigmas = [0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

    for sigma in sigmas:
        print('Sigma:', sigma)

        r_small_sigma = kde_small_sigma.sum_probability(train, validation, sigma)
        r_big_sigma = kde_big_sigma.sum_probability(train, validation, sigma)
        print('Results', r_small_sigma, r_big_sigma)
        assert abs(r_small_sigma - r_big_sigma) < 0.000001


def test_sum_probability_medium():
    np.random.seed(7)

    k = 100
    d = 100
    m = 100

    train = np.random.rand(k, d)
    validation = np.random.rand(m, d)

    sigmas = [0.3, 0.4, 0.5, 0.6, 0.7]

    for sigma in sigmas:
        print('Sigma:', sigma)

        r_small_sigma = kde_small_sigma.sum_probability(train, validation, sigma)
        r_big_sigma = kde_big_sigma.sum_probability(train, validation, sigma)
        print('Results', r_small_sigma, r_big_sigma)
        assert r_small_sigma - r_big_sigma < 0.000001


def test_sum_probability_real_data():
    dir = "../data/"

    train_data = LearningData.from_file(cifar, 'train', dir)

    validation_data = LearningData.from_file(cifar, 'validation', dir)
    print('Validation:', validation_data.shape)

    k = 1000
    m = 100

    train = train_data.features[:k]
    validation = validation_data.features[:m]

    print("Training:", train.shape)
    print('Validation:', validation.shape)

    sigmas = [0.3, 0.4, 0.5, 0.6, 0.7]

    for sigma in sigmas:
        print('Sigma:', sigma)

        r_small_sigma = kde_small_sigma.sum_probability(train, validation, sigma) / m
        r_big_sigma = kde_big_sigma.sum_probability(train, validation, sigma) / m
        print('Results', r_small_sigma, r_big_sigma)
        assert r_small_sigma - r_big_sigma < 0.000001
