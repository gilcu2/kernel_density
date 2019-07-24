from kde_my import *
from learning_data import *


def fit(sigmas: List[float], train: np.ndarray, validation: np.ndarray) -> (float, float):
    better_value = -nan
    better_sigma = -nan
    for sigma in sigmas:
        value = mean_probability(train, validation, sigma)
        print(sigma, value)
        if value > better_value:
            better_sigma = sigma
            better_value = value

    return better_sigma, better_value


if __name__ == '__main__':
    dir = "../data/"
    datas = [mnist, cifar]
    sigmas = [0.05, 0.08, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    sigmas.reverse()

    for data_name in datas:
        train_data = LearningData.from_file(data_name, 'train', dir)
        validation_data = LearningData.from_file(data_name, 'validation', dir)
        fit(sigmas, train_data.features, validation_data.features)
