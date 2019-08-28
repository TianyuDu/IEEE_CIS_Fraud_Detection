"""
Check this script for bridge between data (pre)processing and modelling.
"""
import numpy as np
import pandas as pd

from utils import data_utils
from feature import feature_engineering


if __name__ == "__main__":
    # Load the dataset with nan filled.
    X_train, y_train, X_test = data_utils.load_dataset(path="./data")
    s = "Datasets received: X_train.shape={}; y_train.shape={}, X_test.shape={}"
    print(s.format(X_train.shape, y_train.shape, X_test.shape))

    assert np.all(X_train.index == y_train.index)
    train_index = X_train.index
    test_index = X_test.index

    X_merged = pd.concat([X_train, X_test], axis=0)
    X_merged = feature_engineering.main_fe(X_merged)

    X_train = X_merged.loc[train_index]
    X_test = X_merged.loc[test_index]
    assert np.all(X_train.index == y_train.index)
