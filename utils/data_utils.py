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
import tensorflow as tf

import features


def load_dataset(path: str = "./data") -> pd.DataFrame:
    """
    Loads the dataset from *_forcus.csv.
    NOTE: rename the complete dataset to *_focus.csv to load it.
    """
    # For now, consider transaction dataset only.
    df_train = pd.read_csv(path + "/train_transaction_focus.csv")
    df_test = pd.read_csv(path + "/test_transaction_focus.csv")
    X_train, y_train = _split_data(df_train, data="transaction")
    X_test, y_test = _split_data(df_test, data="transaction")
    return (X_train, y_train), (X_test, y_test)


def _split_data(
    df: pd.DataFrame,
    data: str
) -> Set[pd.DataFrame]:
    """
    Formulates df into a supervised learning (classification) problem.
    Args:
        df: target dataset.
        data: type of data set, either transaction or identity.
    Returns:
        (X, y): feature set and label set, joint by Transaction ID.
        X @ (num_samples, num_features).
        y @ (num_samples, 2) with columns ["TransactionID", "isFraud"].
    """
    if data not in ["transaction", "identity"]:
        raise ValueError("Invalid dataset type")
    if data == "transaction":
        df = features.clean_transaction(df)
    else:
        raise NotImplementedError("Identity dataset is not supported yet.")
    X = df.drop(columns=["TransactionID"])
    y = df[["TransactionID", "isFraud"]]
    print("Positive samples: {}/{} ({:0.4f}%)".format(
        np.sum(y.isFraud), len(y), np.mean(y.isFraud) * 100
    ))
    return X, y


def train_input_fn(X, y, batch_size) -> "TensorSliceDataset":
    """
    Converts the pandas dataset into tensorflow datasets.
    """
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    shuffle_buffer = len(X) // 10  # See doc. of dataset.shuffle.
    # Personally, I don't think it really matter here, as long as we keep
    # the buffer size sufficiently large.
    # Consider training with different buffer size.
    dataset = dataset.shuffle(shuffle_buffer).repeat().batch(batch_size)
    return dataset


def sample_dataset(
    path: str,
    data: str,
    n: Union[int, float] = 0.05,
    random_state: int = 42
) -> None:
    """
    Creates a subsample of the entire training set, to improve performance of model prototyping.
    A {train, test}_{transaction, identity}_focus.csv file
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
