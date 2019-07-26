from kde_big_sigma import sum_probability_parallel
from learning_data import *
from utils import *


def apply(sigma: float, train: np.ndarray, test: np.ndarray):
    m = test.shape[0]

    begin = now()

    print('\nsigma:', sigma, now())

    value = sum_probability_parallel(train, test, sigma) / m

    end = now()
    print('mean log probability:', value)
    print('Time:', end - begin, now())


if __name__ == '__main__':
    dir = "../data/"
    data_names = [mnist, cifar]
    sigma = 0.2

    print('Begin fit', now())

    for data_name in data_names:
        print('\nData:', data_name)

        train_data = LearningData.from_file(data_name, 'train', dir)
        print("Training:", train_data.shape)

        test_data = LearningData.from_file(data_name, 'test', dir)
        print("Test:", test_data.shape)

        apply(sigma, train_data.features, test_data.features)

    print('End apply', now())
