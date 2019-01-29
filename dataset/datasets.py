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
