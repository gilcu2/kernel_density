from kde.input_data import *
import numpy as np

dir = "../data/"

def test_load_mnist():
    data = InputData("mnist", dir)
    assert data.features.shape == (50000, 28, 28)


def test_load_cifar():
    data = InputData(cifar, dir)
    assert data.features.shape == (50000, 32, 32)


def test_to_grey_level():
    rgb = np.arange(12).reshape(3, 2, 2)
    grey = to_grey(rgb)
    assert grey.shape == (2, 2)
    assert grey[0, 0] == 4


def test_vectorizer_to_grey_level():
    rgb_array = np.arange(24).reshape(2, 3, 2, 2)
    print("\nrgb:\n", rgb_array)
    grey_array = np.array([to_grey(image) for image in rgb_array])
    assert grey_array.shape == (2, 2, 2)
    assert grey_array[0, 0, 0] == 4
