"""
Utilities for tensorflow.
"""
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

import tensorflow as tf

from utils import features


def generate_feature_columns(
    X: pd.DataFrame,
    # y: Union[pd.DataFrame, None],
    ID: str = "TransactionID"
) -> List["tf.feature_column"]:
    """
    Generates the correspond feature columns from feature and target datasets.
    Args:
        X: feature dataset for the problem.
        y: target dataset for supervised learning problem.
    Returns:
        (X_fea_col, y_fea_col): feature columns for features and target.
    """
    # ==== y ====
    # y_fea_col = tf.feature_column.categorical_column_with_identity(key="isFraud", num_buckets=2)
    # y_fea_col = tf.feature_column.indicator_column(y_fea_col)
    # ==== X ====
    X_fea_col = []
    for col in X.columns:
        if col == ID:
            continue
        if col in features.CATEGORICAL_TRANS or col in features.CATEGORICAL_ID:
            # Categorical features:
            num_categories = len(set(X[col]))
            # identity_feature_column = tf.feature_column.categorical_column_with_identity(
            #     key=col,
            #     num_buckets=num_categories
            # )
            identity_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key=col,
                vocabulary_list=set(X[col]),
                num_oov_buckets=num_categories
            )
            identity_feature_column = tf.feature_column.indicator_column(identity_feature_column)
            X_fea_col.append(identity_feature_column)
        else:
            # Numerical features:
            numerical_feature_column = tf.feature_column.numeric_column(key=col)
            X_fea_col.append(numerical_feature_column)
    # return X_fea_col, y_fea_col
    return X_fea_col
