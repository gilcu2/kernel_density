from kde.data import *
import numpy as np


def test_load_mnist():
    path = "../data/mnist_py3k.pkl"
    data = Data("mnist", path, (28, 28))
    assert data.features.size == 39200000


def test_load_cifar():
    path = "../data/cifar_batch-1.pkl"
    data = Data("cifar", path, (32, 32))
    assert data.features.size == 30720000


def test_to_fray_level():
    rgb = np.arange(12).reshape(3, 2, 2)
    grey = to_grey(rgb)
    assert grey.shape == (2, 2)
    assert grey[0, 0] == 4


def test_vectorizer_to_fray_level():
    rgb_array = np.arange(24).reshape(2, 3, 2, 2)
    print("\nrgb:\n", rgb_array)
    grey_array = np.array([to_grey(image) for image in rgb_array])
    assert grey_array.shape == (2, 2, 2)
    assert grey_array[0, 0, 0] == 4
