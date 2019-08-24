"""
Aug. 21, 2019
Use this script to store methods that:
i. Manipulate datasets via direct interactions with .csv files on disk.
ii. Spliting and concanating datasets.

NOTE:
Methods in this script do NOT create new columns (features).
"""
from typing import Union, Set
import numpy as np
import pandas as pd

import features


def load_dataset(path: str = "./data") -> pd.DataFrame:
    """
    Loads the dataset from *_forcus.csv.
    NOTE: rename the complete dataset to *_focus.csv to load it.
    """
    df_train = pd.read_csv(path + "/train_transaction_focus.csv")
    df_test = pd.read_csv(path + "/test_transaction_focus.csv")
    X_train, y_train = split_data(df_train)
    X_test, y_test = split_data(df_test)


def split_data(df: pd.DataFrame) -> Set[pd.DataFrame]:
    """
    Formulates df into a supervised learning (classification) problem.
    Returns:
        (X, y): feature set and label set, joint by Transaction ID.
        X @ (num_samples, num_features).
        y @ (num_samples, 2) with columns ["TransactionID", "isFraud"].
    """
    df = features.clean_data(df)
    X = df.drop(columns=["TransactionID"])
    y = df[["TransactionID", "isFraud"]]
    print("Positive samples: {}/{} ({}%)".format(
        np.sum(y.isFraud), len(y), np.mean(y.isFraud)
    ))
    return X, y


def sample_dataset(
    path: str,
    data: str,
    n: Union[int, float] = 0.05,
    random_state: int = 42
) -> None:
    """
    Creates a subsample of the entire training set, to improve performance of model prototyping.
    A [train, test]_[tr, id]_focus.csv file
    containing the sub-sampled dataset will be stored in path provided.

    Args:
        path: data directory.
        n: size of subsample if n >= 1, percentage if n < 1.
        data: which dataset to use, either 'train' or 'test'.
    """
    p = "{}/{}_{}.csv"
    df_trans = pd.read_csv(p.format(path, data, "transaction"))
    # We leave the secondary dataset out for now.
    # df_id = pd.read_csv(p.format(path, data, "identity"))
    if n < 1:
        n = int(n * len(df_trans))
    print("{}/{} {} instances will be sampled.".format(n, len(df_trans), data))
    selected_id = df_trans.sample(n, random_state=42).TransactionID
    mask = df_trans.TransactionID.isin(selected_id)
    df_trans_sub = df_trans[mask].reset_index(drop=True)
    df_trans_sub.to_csv(path + "/{}_{}_focus.csv".format(data, "transaction"), index=False)


if __name__ == "__main__":
    pass
