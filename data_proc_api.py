"""
Check this script for bridge between data (pre)processing and modelling.
"""
import numpy as np
import pandas as pd

from utils import data_utils
from feature import feature_engineering


if __name__ == "__main__":
    # Load the dataset with nan filled.
    (X_train, y_train), (X_test, y_test) = data_utils.load_dataset(path="./data")
    assert np.all(X_train.index == y_train.index)
    train_index = X_train.index
    test_index = X_test.index
    X_all = np.concatenate([X_train, X_test], axis=0)