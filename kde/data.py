from typing import *
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

mnist = 'mnist'
cifar = 'cifar'


def to_grey(image: np.ndarray) -> np.ndarray:
    sum = image.sum(axis=0)
    grey = sum / 3
    return grey


def to_normalize(image: np.ndarray) -> np.ndarray:
    sum = image.sum(axis=0)
    grey = sum / 255
    return grey


class Data:

    def __init__(self, name: str, path: str, size: Tuple[int, int]) -> None:
        self.name = name
        self.size = size
        self.features, self.labels = self.load(path, name)

    @staticmethod
    def load(path: str, name: str) -> Tuple[np.ndarray, np.ndarray]:
        file = open(path, 'rb')
        data = pickle.load(file, encoding='bytes')
        file.close()
        if name == mnist:
            features, labels = data[0]
            features = features.reshape(50000, 28, 28)
            return features, labels
        elif name == cifar:
            labels = np.asarray(data[b"labels"])
            rgb_array = data[b"data"].reshape(10000, 3, 32, 32)
            features = np.array([to_grey(image) for image in rgb_array])
            features = np.array([to_normalize(image) for image in features])
            return features, labels

    def show(self) -> None:
        print("label: ", self.labels[0])
        plt.imshow(self.features[0], cmap=cm.Greys_r)
        plt.show()


if __name__ == '__main__':
    path = "../data/mnist_py3k.pkl"
    data = Data("mnist", path, (28, 28))
    data.show()

    path = "../data/cifar_batch-1.pkl"
    data = Data(cifar, path, (32, 32))
    data.show()
