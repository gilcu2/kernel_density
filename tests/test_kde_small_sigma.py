from time import time

from kde.kde_small_sigma import *


def test_square_difference():
    v1 = np.array([0.1, 0.2])
    v2 = np.array([0.4, 0.6])
    square_diff = square_difference(v1, v2)
    assert square_diff == 0.25


def test_square_difference_times():
    begin = time()
    v1 = np.random.rand(10000000)
    v2 = np.random.rand(10000000)
    # np.dot(v1, v2)
    square_diff = square_difference(v1, v2)
    total = time() - begin
    print("Time:", total)

    assert total < 1


def test_logpx():
    x = np.array([0.1, 0.2])
    mat = np.array([
        [0.1, 0.2],
        [0.2, 0.3],
    ])
    r = logpx(x, mat, 0, 1, 0)
    assert r > 0.0


def test_logpx():
    train = np.array([
        [0.1, 0.2],
        [0.2, 0.3],
    ])
    validation = np.array([
        [0.1, 0.3],
        [0.2, 0.7],
    ])
    r = mean_probability(train, validation, sigma=1.0)
    assert r < 0.0


def test_sum_probabilities_samples():
    train = np.array([
        [0.1, 0.2],
        [0.2, 0.3],
    ])
    validation = np.array([
        [0.1, 0.3],
        [0.2, 0.7],
    ])

    sigma = 0.1

    r = sum_probability(train, validation, sigma)
    print("sum:", r)
    assert r < 0.0


def test_sum_probabilities_generated():
    np.random.seed(7)

    k = 1
    m = 1
    d = 200

    train = np.random.rand(k, d)
    validation = np.random.rand(m, d)

    sigma = 0.05

    r = sum_probability(train, validation, sigma)
    print("sum:", r)
    assert r < 0.0


def test_sum_probability_real_data():
    dir = "../data/"

    train_data = LearningData.from_file(cifar, 'train', dir)
    print("Training:", train_data.shape)

    validation_data = LearningData.from_file(cifar, 'validation', dir)
    print("Validation:", validation_data.shape)

    k = 1
    m = 1

    train = train_data.features[:k]
    validation = validation_data.features[:m]

    sigma = 0.1

    r_propoused = sum_probability(train, validation, sigma)
    assert r_propoused + 10 < 0.000001
