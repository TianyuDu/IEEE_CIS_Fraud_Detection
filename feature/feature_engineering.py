"""
This script contains feature engineering for this project.
"""
import numpy as np
import pandas as pd

from utils import data_utils
from utils import feature_utils


def modify_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    All in one method for feature engineering.
    """
    df = df.copy()
    df = pca_and_cluster(df, columns="V", n_components=35, n_clusters=6)
    return df