from kde.data import Data


def test_load_mnist():
    path = "../data/mnist_py3k.pkl"
    data = Data("mnist", path, (28, 28))
    assert data.features.size == 39200000


def test_load_cifar():
    path = "../data/cifar_batch-1.pkl"
    data = Data("cifar", path, (32, 32))
    assert data.features.size == 30720000
