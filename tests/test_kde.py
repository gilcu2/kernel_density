from kde.kde import *
import numpy as np


def test_reduce_vector():
    zigma = 1.0
    c1 = 2 * zigma * zigma
    c2 = 0.5 * log(pi * c1)
    xj = 0.0
    vector = np.arange(4.0)
    c3 = 1.0 / vector.size
    r = reduce_vector(xj, vector, c1, c2, c3)
    assert r < 0.0


def test_logpx():
    zigma = 1.0
    c1 = 2 * zigma * zigma
    c2 = 0.5 * log(pi * c1)
    x = np.arange(4.0)
    mat = np.arange(8.0).reshape(2, 4)
    c3 = 1.0 / mat.shape[0]
    r = logpx(x, mat, c1, c2, c3)
    assert r < 0.0
