from kde.fit import *


def test_fit_time():
    dir = "../data/"
    name = 'mnist'
    sigmas = [0.05]
    begin = time()
    train_data = LearningData.from_file(name, 'train', dir)
    validation_data = LearningData.from_file(name, 'validation', dir)
    fit(sigmas, train_data.features, validation_data.features)

    total = time() - begin
    print("Time:", total)

    assert total < 100
