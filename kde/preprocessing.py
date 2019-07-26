from input_data import *

if __name__ == '__main__':
    dir = "../data/"
    datas = [mnist, cifar]

    for data_name in datas:
        data = InputData(data_name, dir)
        data.shuffle()
        # data.show()
        data.save_learning(dir)
        print('Data', data_name, 'saved')
