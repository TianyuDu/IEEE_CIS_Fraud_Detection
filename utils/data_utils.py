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

import utils.feature_utils as feature_utils
from utils import ram_utils


def load_feature_set(
    path: str = "./data"
) -> Set[pd.DataFrame]:
    """
    ...
    """
    raise NotImplementedError


def generate_feature_set(
    src_path: str = "./data",
    dest_path: str = "./data"
) -> None:
    """
    ...
    """
    raise NotImplementedError


def load_dataset(
    path: str = "./data",
    reduce_ram: bool = False
) -> Set[pd.DataFrame]:
    """
    Loads the dataset from *_forcus.csv.
    NOTE: rename the complete dataset to *_focus.csv to load it.
    """
    # For now, consider transaction dataset only.
    # Checked: TransactionIDs are all unique.
    df_train = pd.read_csv(path + "/train_transaction.csv", index_col="TransactionID")
    df_test = pd.read_csv(path + "/test_transaction.csv", index_col="TransactionID")
    if reduce_ram:
        df_train = ram_utils.reduce_mem_usage(df_train)  # Optional.
        df_test = ram_utils.reduce_mem_usage(df_test)
    X_train, y_train = _split_data(df_train)
    X_test = df_test
    assert X_train.shape[1] == X_test.shape[1]

    X_train, X_test = feature_utils.clean_transaction_2(X_train, X_test)

    return X_train, y_train, X_test


def _split_data(
    df: pd.DataFrame
) -> Set[pd.DataFrame]:
    print("Creating feature and target datasets...")
    X = df.drop(columns=["isFraud"])
    y = df[["isFraud"]]

    print("Positive samples: {}/{} ({:0.4f}%)".format(
        np.sum(y.isFraud), len(y), np.mean(y.isFraud) * 100
    ))
    print("X.shape={}, y.shape={}".format(X.shape, y.shape))
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


def generate_submission(
    prob: Union[np.ndarray, pd.DataFrame],
    dest_path: str = "./submission.csv",
    src_path: str = "./data"
) -> None:
    """
    Generates predicted probabilities for submission.
    """
    sample_submission = pd.read_csv(
        src_path + "/sample_submission.csv",
        index_col="TransactionID"
    )

    if type(prob) is pd.DataFrame:
        assert np.all(
            sample_submission.index == prob.index
        )
        holder = prob.copy()
    elif type(prob) is np.ndarray:
        assert prob.shape[0] == len(sample_submission)
        holder = sample_submission.copy()
        holder["isFraud"] = prob

    holder.to_csv(dest_path, header=True, index=True)
