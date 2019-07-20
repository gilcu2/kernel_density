from typing import *
import gzip
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

LearnMatrix = Tuple[np.ndarray, np.ndarray]


class Data:

    def __init__(self, name: str, path: str, size: Tuple[int, int]) -> None:
        self.name = name
        self.size = size
        self.train_x, self.train_y = self.load(path)

    @staticmethod
    def load(path: str) -> LearnMatrix:
        file = open(path, 'rb')
        data = pickle.load(file, encoding='bytes')
        if type(data) == tuple:
            return data[1]
        else:
            return data

    def show(self) -> None:
        plt.imshow(self.train_x[0].reshape(self.size), cmap=cm.Greys_r)
        plt.show()


if __name__ == '__main__':
    path = "../data/mnist_py3k.pkl.gz"
    data = Data("mnist", path, (28, 28))
    data.show()
