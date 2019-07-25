from math import *

import kde_small_sigma
from learning_data import *


def fit(sigmas: List[float], train: np.ndarray, validation: np.ndarray) -> (float, float):
    better_value = -nan
    better_sigma = -nan
    m = validation.shape[0]
    for sigma in sigmas:
        begin = now()

        print('\nsigma:', sigma, now())

        value = kde_small_sigma.sum_probability_parallel(train, validation, sigma) / m

        end = now()
        print('mean log probability:', value)
        print('Time:', end - begin, now())

        if value > better_value:
            better_sigma = sigma
            better_value = value

    return better_sigma, better_value


if __name__ == '__main__':
    dir = "../data/"
    datas = [mnist, cifar]
    sigmas = [0.08, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    sigmas.reverse()

    print('Begin fit', now())

    for data_name in datas:
        print('Data:', data_name)

        train_data = LearningData.from_file(data_name, 'train', dir)
        print("Training:", train_data.shape)

        validation_data = LearningData.from_file(data_name, 'validation', dir)
        print("Validation:", validation_data.shape)

        fit(sigmas, train_data.features, validation_data.features)

    print('End fit', now())
