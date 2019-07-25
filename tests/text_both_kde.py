import kde_proposed
import kde_my
import numpy as np


def test_sum_probability():
    train = np.array([
        [0.1, 0.2],
        [0.2, 0.3],
    ])
    validation = np.array([
        [0.1, 0.3],
        [0.2, 0.7],
    ])
    r_propoused = kde_proposed.sum_probability(train, validation, sigma=1.0)
    r_main = kde_my.sum_probability(train, validation, sigma=1.0)
    assert r_propoused == r_main
