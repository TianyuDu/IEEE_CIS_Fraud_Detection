"""
This script contains feature engineering for this project.
"""
import numpy as np
import pandas as pd

from utils import data_utils
from utils import feature_utils


def main_fe(df: pd.DataFrame) -> pd.DataFrame:
    """
    All in one method for feature engineering.
    """
    df = df.copy()
    # Modify numerical features.
    df = feature_utils.pca_and_cluster(df, columns="V", n_components=35, n_clusters=6)
    df = feature_utils.pca_and_cluster(df, columns="C", n_components=3, n_clusters=4)
    df = feature_utils.pca_and_cluster(df, columns="D", n_components=3, n_clusters=8)
    return df
