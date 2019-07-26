import csv
from math import *

from kde_big_sigma import sum_probability_parallel
from learning_data import *

FitResult = Tuple[float, float, datetime.timedelta]
FitResults = Tuple[float, float, Sequence[FitResult]]


def fit(sigmas: List[float], train: np.ndarray, validation: np.ndarray) -> FitResults:
    better_value = -nan
    better_sigma = -nan
    m = validation.shape[0]
    results = []
    for sigma in sigmas:
        begin = now()

        print('\nsigma:', sigma, now())

        value = sum_probability_parallel(train, validation, sigma) / m

        end = now()
        total_time = end - begin
        print('mean log probability:', value)
        print('Time:', total_time, now())
        results.append((sigma, value, total_time))

        if value > better_value:
            better_sigma = sigma
            better_value = value

    print("Better:", sigma, value)

    return better_sigma, better_value, results


def save_results(all_results: Dict[str, FitResults]):
    def save_csv(data: List[Tuple[Any]], path: str):
        with open(path, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(data)

        csvFile.close()

    def save_csv_results(name: str, results: FitResults):
        csv_data = [['sigma', 'quality', 'time']]
        for r in results:
            csv_data.append(r)
        save_csv(name + '-fit.csv')

    best = [['data', 'sigma', 'quality']]
    for (data_name, results) in all_results:
        best.append([data_name, results[0], results[1]])
        save_csv_results(data_name, results[2])
    save_csv(best, 'kde-fit.csv')


if __name__ == '__main__':
    dir = "../data/"
    datas = [mnist, cifar]
    sigmas = [0.05, 0.08, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

    print('Begin fit', now())

    results = {}

    for data_name in datas:
        print('\nData:', data_name)

        train_data = LearningData.from_file(data_name, 'train', dir)
        print("Training:", train_data.shape)

        validation_data = LearningData.from_file(data_name, 'validation', dir)
        print("Validation:", validation_data.shape)

        result = fit(sigmas, train_data.features, validation_data.features)
        results[data_name] = result

    save_results(results)

    print('End fit', now())
