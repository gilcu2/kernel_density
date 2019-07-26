#!/usr/bin/env sh

echo getting mnist
wget http://www-labs.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
gzip -d mnist.pkl.gz

echo getting ciffar
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz

mv cifar-10-batches-py/data_batch_1 cifar-1.pkl
mv cifar-10-batches-py/data_batch_2 cifar-2.pkl
mv cifar-10-batches-py/data_batch_3 cifar-3.pkl
mv cifar-10-batches-py/data_batch_4 cifar-4.pkl
mv cifar-10-batches-py/data_batch_5 cifar-5.pkl
mv cifar-10-batches-py/test_batch cifar-test.pkl

echo data downloaded


