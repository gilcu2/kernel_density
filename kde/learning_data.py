from typing import *
import input_data
from input_data import mnist, cifar, text_labels, show
import numpy as np
import pickle


def load_pickle(path: str) -> Any:
    file = open(path, 'rb')
    data = pickle.load(file, encoding='bytes')
    file.close()
    return data


class LearningData:

    def __init__(self, name: str, features: np.ndarray, labels: np.ndarray, text_labels: dict) -> None:
        self.name = name
        self.features = features
        self.size = self.features[0].shape
        self.labels = labels
        self.text_labels = text_labels

    @classmethod
    def from_file(cls, name: str, dir: str):
        text_labels = input_data.text_labels[name]

        path = dir + name + '-train-features.pkl'
        features = load_pickle(path)

        path = dir + name + '-train-labels.pkl'
        labels = load_pickle(path)
        return LearningData(name, features, labels, text_labels)

    def show(self):
        show(self)

    def get_label(self, i) -> str:
        return self.text_labels[self.labels[i]]

    def get_one_class(self, class_id: int):
        indices = np.where(self.labels == class_id)
        labels = self.labels[indices]
        features = self.features[indices]
        return LearningData(self.name, features, labels, self.text_labels)


if __name__ == '__main__':
    dir = "../data/"
    datas = [mnist, cifar]

    #
    # for data_name in datas:
    #     data = LearningData(data_name, dir)
    #     data.show()

    data = LearningData.from_file(mnist, dir)
    ceros = data.get_one_class(3)
    ceros.show()