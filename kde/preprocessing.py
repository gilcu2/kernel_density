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
    return image / 255


class Data:

    def __init__(self, name: str, dir: str) -> None:
        self.name = name
        if name == mnist:
            self.load_mnist(dir)
        elif name == cifar:
            self.load_cifar(dir)

        print(self.name, self.features.shape)

    def load_mnist(self, dir: str):
        self.size = (28, 28)
        path = dir + mnist + '.pkl'
        file = open(path, 'rb')
        data = pickle.load(file, encoding='bytes')
        file.close()
        self.features, self.labels = data[0]
        self.features = self.features.reshape(50000, 28, 28)
        self.human_labels = [str('i') for i in range(9)]

    def load_cifar(self, dir: str):
        self.size = (32, 32)
        labels_files = []
        features_files = []
        for i in range(5):
            path = dir + cifar + '-' + str(i + 1) + '.pkl'
            file = open(path, 'rb')
            labels, features = self.load_cifar_file(file)
            labels_files.append(labels)
            features_files.append(features)
        self.labels = np.concatenate(labels_files)
        self.features = np.concatenate(features_files)

    def load_cifar_file(self, file):
        data = pickle.load(file, encoding='bytes')
        file.close()
        labels = np.asarray(data[b"labels"])
        rgb_array = data[b"data"].reshape(10000, 3, 32, 32)
        features = np.array([to_grey(image) for image in rgb_array])
        features = np.array([to_normalize(image) for image in features])
        return labels, features

    def show(self) -> None:
        print("label: ", self.labels[0])
        plt.imshow(self.features[0], cmap=cm.Greys_r)
        plt.show()


if __name__ == '__main__':
    dir = "../data/"

    data = Data(mnist, dir)
    data.show()

    data = Data(cifar, dir)
    data.show()
