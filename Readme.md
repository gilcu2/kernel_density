# Kernel Density Estimator 

## Steps

### Unzip

- Unzip the answer file
- Report in docs/kde.pdf

### Get the data

- cd data
- ./download.sh
- After that must be in data dir: mnist.pkl,cifar-[1..5].pkl and cifar-test.pkl

### Install python env and dependencies (Require python3 and pip in the path)
- cd ..
- ./venv.sh

### Software Test
- cd ../tests
- pytest
- All test must end ok

### Data preprocessing

- cd ../kde
- ./preprocessing.py
- When finish must have mnist-[train,validation,test].pkl and cifar-[train,validation,test].pkl in data dir

### kde fit
- ./fit.py
- when finish must have kde-fit.csv, mnist-fit.csv and cifar-fit.csv in dta dir

# kde test
- ./apply.py
- When finish must have kde-results.csv in data dir
