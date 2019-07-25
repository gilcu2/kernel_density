import pickle
from datetime import datetime
from typing import *

import numpy as np

import input_data
from input_data import mnist, cifar, show


def now():
    return datetime.now()


def load_pickle(path: str) -> Any:
    file = open(path, 'rb')
    data = pickle.load(file, encoding='bytes')
    file.close()
    return data


class LearningData:

    def __init__(self, name: str, features: np.ndarray, labels: np.ndarray, text_labels: dict) -> None:
        self.name = name
        self.shape = features.shape
        self.features = features.reshape(self.shape[0], self.shape[1] * self.shape[2])
        self.size = self.shape[0]
        self.labels = labels
        self.text_labels = text_labels

    @classmethod
    def from_file(cls, name: str, type: str, dir: str):
        text_labels = input_data.text_labels[name]

        path = dir + name + '-' + type + '-features.pkl'
        features = load_pickle(path)

        path = dir + name + '-' + type + '-labels.pkl'
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

    for data_name in datas:
        data = LearningData.from_file(data_name, dir)
        data.show()
