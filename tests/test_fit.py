from kde.fit import *


def test_fit_time():
    dir = "../data/"
    name = 'mnist'
    sigmas = [0.05]
    k = 1000
    m = 1000
    begin = now()
    train = LearningData.from_file(name, 'train', dir).features[:k]
    validation = LearningData.from_file(name, 'validation', dir).features[:m]

    print('train:', train.shape)
    print('validation:', validation.shape)
    fit(sigmas, train, validation)

    total = now() - begin
    print("Time:", total)

    assert total.total_seconds() < 1
