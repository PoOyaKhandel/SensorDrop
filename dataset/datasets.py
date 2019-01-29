import numpy as np
import os
from sklearn.model_selection import train_test_split


def get_mvmc(te_percent=0.25):
    """"
    pre-process dataset
    """
    base_dir = "D:\Library\Statistical Learning\SensorDrop\dataset"
    path = os.path.join(base_dir, 'mvmc.npz')

    data = np.load(path)
    X = data['X']
    y = data['y']

    return split_labels(*train_test_split(X, y, test_size=te_percent))


def split_labels(x_train, x_test, y_train, y_test):
    """"
    split each device train/test set into dictionary arguments
    """
    xi_tr = {}
    yi_tr = {}
    xi_te = {}
    yi_te = {}

    for i in range(6):
        xi_tr[str(i)] = x_train[:, i, :, :, :]
        xi_te[str(i)] = x_test[:, i, :, :, :]
        yi_tr[str(i)] = y_train[:, i]
        yi_te[str(i)] = y_test[:, i]

    return xi_tr, xi_te, yi_tr, yi_te


def get_mvmc_concat(te_percent=0.25):
    base_dir = "D:\Library\Statistical Learning\SensorDrop\dataset"
    path = os.path.join(base_dir, 'mvmc.npz')

    data = np.load(path)
    X = data['X']
    y = data['y']

    return concat_split_labels(*train_test_split(X, y, test_size=te_percent))


def concat_split_labels(x_train, x_test, y_train, y_test):

    x_train = np.reshape(x_train, (x_train.shape[0]*x_train.shape[1], x_train.shape[2], x_train.shape[3],
                                   x_train.shape[4]))

    x_test = np.reshape(x_test, (x_test.shape[0]*x_test.shape[1], x_test.shape[2], x_test.shape[3],
                                 x_test.shape[4]))

    y_train = np.reshape(y_train, (y_train.shape[0]*y_train.shape[1], 1))

    y_test = np.reshape(y_test, (y_test.shape[0]*y_test.shape[1], 1))


    return x_train, x_test, y_train, y_test