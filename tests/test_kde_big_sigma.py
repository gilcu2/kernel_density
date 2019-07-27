from time import time

from kde.kde_big_sigma import *


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


def test_LSE():
    v = np.arange(100)
    direct = np.log(np.sum(np.exp(v)))
    lse = LSE(v)
    assert lse - direct < 0.000000001


def test_log_sum1():
    x = np.array([[1.0, 1.0]])
    y = np.array([[2.0, 1.0]])
    r = log_sum(x, y, 1.0)
    expected = -1.0
    print(r, expected)
    assert r == expected


def test_log_sum2():
    mat = np.array([
        [1.0, 1.0],
        [3.0, 1.0],
    ])
    vect = np.array([2.0, 1.0])

    r = log_sum(mat, vect, 1.0)
    expected = log(2) - 1
    print(r, expected)
    assert abs(r - expected) < 0.0000001


def test_dot_mat_vec():
    mat = np.array([
        [1.0, 2.0],
        [3.0, -1.0],
    ])
    vect = np.array([1.0, 2.0])

    r = np.dot(mat, vect)
    expected = np.array([np.dot(mat[0], vect), np.dot(mat[1], vect)])
    print(r, expected)
    assert np.array_equal(r, expected)


def test_diff_mat_vec():
    mat = np.array([
        [1.0, 2.0],
        [3.0, -1.0],
    ])
    vect = np.array([1.0, 2.0])

    r = mat - vect
    expected = np.array([mat[0] - vect, mat[1] - vect])
    print(r, expected)
    assert np.array_equal(r, expected)


def test_vecs_norm2():
    mat = np.array([
        [1.0, 2.0],
        [3.0, -1.0],
    ])

    norms2 = np.square(np.linalg.norm(mat, axis=1))

    expected = np.array([5.0, 10.0])
    print(norms2, expected)
    assert np.linalg.norm(norms2 - expected) < 0.000001

def test_sum_probability_parallel_minimun():
    train = np.array([
        [0.1, 0.1],
        [0.4, 0.1],
    ])
    validation = np.array([
        [0.2, 0.1],
        # [0.2, 0.7],
    ])

    sigma = 1.0

    r_sequential = sum_probability(train, validation, sigma)
    r_parallel = sum_probability_parallel(train, validation, sigma)
    print('Results', r_sequential, r_parallel)
    assert abs(r_sequential - r_parallel) < 0.0001

def test_sum_probability_paralle_small_data():
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

        r_sequential = sum_probability(train, validation, sigma)
        r_parallel = sum_probability_parallel(train, validation, sigma)
        print('Results', r_sequential, r_parallel)
        assert abs(r_sequential - r_parallel) < 0.00001

def test_sum_probability_parallel_medium():
    np.random.seed(7)

    k = 100
    d = 100
    m = 100

    train = np.random.rand(k, d)
    validation = np.random.rand(m, d)

    sigmas = [0.3, 0.4, 0.5, 0.6, 0.7]

    for sigma in sigmas:
        print('Sigma:', sigma)

        r_sequential = sum_probability(train, validation, sigma)
        r_parallel = sum_probability_parallel(train, validation, sigma)
        print('Results', r_sequential, r_parallel)
        assert abs(r_sequential - r_parallel) < 0.00001

def test_sum_probability_parallel_real_data():
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

        r_sequential = sum_probability(train, validation, sigma)
        r_parallel = sum_probability_parallel(train, validation, sigma)
        print('Results', r_sequential, r_parallel)
        assert abs(r_sequential - r_parallel) < 0.00001
