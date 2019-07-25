from kde.kde_big_sigma import *
from time import time


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

