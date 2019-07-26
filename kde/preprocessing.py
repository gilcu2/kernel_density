#!/usr/bin/env python3

import sys

sys.path.extend(['../', '../kde'])

from kde.input_data import *

if __name__ == '__main__':
    dir = "../data/"
    data_names = [mnist, cifar]

    for data_name in data_names:
        data = InputData(data_name, dir)
        data.shuffle()
        # data.show()
        data.save_learning(dir)
        print('Data', data_name, 'saved')
