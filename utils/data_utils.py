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
from utils import mem_utils


def load_feature_set(
    path: str = "./data",
    df_with_features: str = "df_with_features.csv"
) -> Set[pd.DataFrame]:
    """
    Load pre-made dataset with features.
    """
    df = pd.read_csv(path + "/" + df_with_features, index_col="TransactionID")
    print("Featured dataset loaded @ {}".format(df.shape))
    raw_train = pd.read_csv(path + "/train_transaction.csv", index_col="TransactionID")
    raw_test = pd.read_csv(path + "/test_transaction.csv", index_col="TransactionID")

    df_train = df.loc[raw_train.index]
    df_test = df.loc[raw_test.index]
    assert df_train.shape[0] + df_test.shape[0] == df.shape[0]

    X_train = df_train.drop(columns=["isFraud"])
    y_train = df_train["isFraud"]
    X_test = df_test.drop(columns=["isFraud"])
    print("Extracted: X_train @ {}, X_test @ {}".format(X_train.shape, X_test.shape))

    return X_train, y_train, X_test


def save_feature_set(
    path: str = "./data/saved_features.csv",
    reduce_mem: bool = False
) -> None:
    """
    Save created features to local disk.
    """
    X_train, y_train, X_test = load_dataset(path="./data", reduce_mem=False)
    if reduce_mem:
        X_train = mem_utils.reduce_mem_usage(X_train)
        X_test = mem_utils.reduce_mem_usage(X_test)
    df_train = pd.concat([y_train, X_train], axis=1)
    print("df_train.shape: {}".format(df_train.shape))

    y_test_placeholder = pd.DataFrame(
        data=["test"] * X_test.shape[0],
        index=X_test.index,
        columns=["isFraud"])
    df_test = pd.concat([
        y_test_placeholder, X_test
    ], axis=1)
    print("df_test.shape: {}".format(df_test.shape))

    assert np.all(df_train.columns == df_test.columns)

    df_all = pd.concat([df_train, df_test])
    print("df_all.shape: {}".format(df_all.shape))
    if reduce_mem:
        df_all = mem_utils.reduce_mem_usage(df_all)
    try:
        df_all.to_csv(path, index=True, header=True)
    except FileNotFoundError:
        print("The path provided does not exist: {}".format(path))
        print("Featured dataset is saved to: ./temp_feature_map.csv")
        df_all.to_csv("./temp_feature_map.csv", index=True, header=True)


def load_dataset(
    path: str = "./data",
    reduce_mem: bool = False
) -> Set[pd.DataFrame]:
    """
    Loads the dataset from *_forcus.csv.
    NOTE: rename the complete dataset to *_focus.csv to load it.
    """
    # For now, consider transaction dataset only.
    # Checked: TransactionIDs are all unique.
    print("Loading training set...")
    df_train = pd.read_csv(path + "/train_transaction.csv", index_col="TransactionID")
    print("Loading testing set...")
    df_test = pd.read_csv(path + "/test_transaction.csv", index_col="TransactionID")
    if reduce_mem:
        print("Compressing dataframes...")
        df_train = mem_utils.reduce_mem_usage(df_train)  # Optional.
        df_test = mem_utils.reduce_mem_usage(df_test)
    print("Spliting data...")
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
