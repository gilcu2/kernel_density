from time import time

from kde.kde_big_sigma import *


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


def test_LSE():
    v = np.arange(1000)
    direct = np.log(np.sum(np.exp(v)))
    lse = LSE(v)
    assert lse - direct < 0.000000001


def test_log_sum1():
    x = np.array([1.0, 1.0])
    y = np.array([2.0, 1.0])
    r = log_sum(x, y, 1.0)
    expected = 1.0
    print(r, expected)
    assert r == expected


def test_log_sum2():
    mat = np.array([
        [1.0, 1.0],
        [3.0, 1.0],
    ])
    vect = np.array([2.0, 1.0])

    r = log_sum(mat, vect, 1.0)
    expected = log(2 * e)
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
