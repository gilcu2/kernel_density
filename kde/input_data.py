import pickle
import numpy as np
import matplotlib.pyplot as plt

mnist = 'mnist'
cifar = 'cifar'

mnist_text_labels = [str(i) for i in range(10)]
cifar_text_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

text_labels = {mnist: mnist_text_labels, cifar: cifar_text_labels}


def to_grey(image: np.ndarray) -> np.ndarray:
    sum = image.sum(axis=0)
    grey = sum / 3
    return grey


def to_normalize(image: np.ndarray) -> np.ndarray:
    return image / 255


def show(data, rows: int = 4, columns: int = 4) -> None:
    for i in range(rows * columns):
        image = data.features[i].reshape(data.shape[1], data.shape[2])
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(data.get_label(i))
        plt.axis('off')  # do not show axis value
    plt.tight_layout()
    plt.show()


class InputData:

    def __init__(self, name: str, dir: str) -> None:
        self.name = name
        if name == mnist:
            self.load_mnist(dir)
        elif name == cifar:
            self.load_cifar(dir)

        print(self.name, self.features.shape)

    def shuffle(self) -> None:
        rng_state = np.random.get_state()
        np.random.shuffle(self.labels)
        np.random.set_state(rng_state)
        np.random.shuffle(self.features)

    def load_mnist(self, dir: str):
        self.size = (28, 28)
        path = dir + mnist + '.pkl'
        file = open(path, 'rb')
        data = pickle.load(file, encoding='bytes')
        file.close()

        self.features, self.labels = data[0]
        self.features = self.features.reshape(50000, 28, 28)
        self.text_labels = mnist_text_labels

        self.test_features, self.test_labels = data[2]
        self.test_features = self.test_features.reshape(10000, 28, 28)

    def load_cifar(self, dir: str):
        self.size = (32, 32)
        self.load_cifar_train(dir)
        self.load_cifar_test(dir)
        self.text_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def load_cifar_train(self, dir):
        labels_files = []
        features_files = []
        for i in range(5):
            path = dir + cifar + '-' + str(i + 1) + '.pkl'
            file = open(path, 'rb')
            labels, features = self.load_cifar_file(file)
            file.close()
            labels_files.append(labels)
            features_files.append(features)
        self.labels = np.concatenate(labels_files)
        self.features = np.concatenate(features_files)

    def load_cifar_test(self, dir):
        path = dir + cifar + '-' + 'test' + '.pkl'
        file = open(path, 'rb')
        self.test_labels, self.test_features = self.load_cifar_file(file)
        file.close()


    def load_cifar_file(self, file):
        data = pickle.load(file, encoding='bytes')
        labels = np.asarray(data[b"labels"])
        rgb_array = data[b"data"].reshape(10000, 3, 32, 32)
        features = np.array([to_grey(image) for image in rgb_array])
        features = np.array([to_normalize(image) for image in features])
        return labels, features

    def get_label(self, i) -> str:
        return self.text_labels[self.labels[i]]

    def show(self):
        show(self)

    def save_learning(self, dir: str) -> None:
        labels_splits = np.split(self.labels, 5)
        features_splits = np.split(self.features, 5)

        pickle.dump(labels_splits[0], open(dir + self.name + '-train-labels.pkl', "bw"))
        pickle.dump(features_splits[0], open(dir + self.name + '-train-features.pkl', "bw"))

        pickle.dump(labels_splits[1], open(dir + self.name + '-validation-labels.pkl', "bw"))
        pickle.dump(features_splits[1], open(dir + self.name + '-validation-features.pkl', "bw"))

        pickle.dump(self.test_labels, open(dir + self.name + '-test-labels.pkl', "bw"))
        pickle.dump(self.test_features, open(dir + self.name + '-test-features.pkl', "bw"))

        print(self.name, 'split', labels_splits[0].shape, features_splits[0].shape)


if __name__ == '__main__':
    dir = "../data/"
    datas = [mnist, cifar]

    for data_name in datas:
        data = InputData(data_name, dir)
        data.shuffle()
        data.show()
