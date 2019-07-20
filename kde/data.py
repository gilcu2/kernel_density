from typing import *
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class Data:

    def __init__(self, name: str, path: str, size: Tuple[int, int]) -> None:
        self.name = name
        self.size = size
        self.features, self.labels = self.load(path)

    @staticmethod
    def load(path: str) -> Tuple[np.ndarray, np.ndarray]:
        file = open(path, 'rb')
        data = pickle.load(file, encoding='bytes')
        file.close()
        if type(data) == tuple:
            features, labels = data
            features = features.reshape(28, 28)
            return features, labels
        else:
            labels = np.asarray(data[b"labels"])
            features = data[b"data"].reshape(10000, 3, 32, 32)
            return features, labels

    def show(self) -> None:
        plt.imshow(self.features[0], cmap=cm.Greys_r)
        plt.show()


if __name__ == '__main__':
    # path = "../data/mnist_py3k.pkl"
    # data = Data("mnist", path, (28, 28))
    # data.show()

    path = "../data/cifar_batch-1.pkl"
    data = Data("cifar", path, (32, 32))
    data.show()
