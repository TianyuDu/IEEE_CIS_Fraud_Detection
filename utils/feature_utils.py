"""
Aug. 21, 2019
Use this script to store methods that:
i. Manipulate coulumns of dataset, like creating new features.
"""
from typing import List, Union

import numpy as np
import pandas as pd

from sklearn import decomposition
from sklearn import preprocessing
from sklearn import cluster


# Categorical columns in transaction dataset.
CATEGORICAL_TRANS = sum(
    [
        ["card" + str(i) for i in range(1, 7)],
        ["addr1", "addr2"],
        ["P_emaildomain", "R_emaildomain"],
        ["M" + str(i) for i in range(1, 10)]
    ],
    ["ProductCD"]
)

# How many categories left after pruning categorical variables.
# TODO: this should be determined from EDA.
PRUNE_DICT_TRANS = {
    "P_emaildomain": 10,
}

# Categorical columns in identity dataset.
CATEGORICAL_ID = ["DeviceType", "DeviceInfo"] + [
    "id_" + str(i) for i in range(12, 39)
]


def clean_transaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the transaction dataset.
    """
    # Copy dataframe to local, so that the original dataset is not modified.
    # See /notes/copy_df.py for an example.
    df = df.copy()
    # Drop features with insufficient observations.
    df = _drop_inferior_features_transaction(df, nan_threshold=0.3)
    # Mask missing nan observations left.
    df = _fill_nan(df, categorical_fill="Missing", numerical_fill=-1)
    df = _prune_categories(df, prune_dict=PRUNE_DICT_TRANS)
    df = _convert_to_dummies(df)
    return df


def clean_transaction_2(X_train, X_test):
    """
    Cleans the transaction dataset, supporting two datasets.
    """
    # Concanate training and testing set.
    train_index = X_train.index
    test_index = X_test.index

    merged = pd.concat([X_train, X_test], axis=0)
    print("Merged dataset (train + test): {}".format(merged.shape))
    cleaned = clean_transaction(merged)

    X_train = cleaned.loc[train_index]
    X_test = cleaned.loc[test_index]
    print("Cleaned X_train.shape={}".format(X_train.shape))
    print("Cleaned X_test.shape={}".format(X_test.shape))
    return X_train, X_test


def _drop_inferior_features_transaction(
    df: pd.DataFrame,
    nan_threshold: float,
    target: str = "isFraud"
) -> pd.DataFrame:
    """
    Drop inferior features (e.g. those with too many nan values) in the
    transaction dataset.
    Args:
        df:
            The source dataframe.
        nan_threshold:
            If the percentage of nan observations in one feature
            column was beyond his threshold, this column would be dropped.
        target:
            The name of target column, this column will be perserved.
    """
    print("Executing inferior feature removal...")
    df = df.copy()
    num_columns = df.shape[1]
    if nan_threshold > 1.0 or nan_threshold < 0.0:
        raise ValueError("nan_threshold should be in range [0, 1].")

    for col in df.columns:
        if col == target:  # Preserve the target column.
            continue
        nan_percentage = np.mean(df[col].isna())
        if nan_percentage >= nan_threshold:
            df.drop(columns=[col], inplace=True)
    print("{}/{} features left with nan threshold {}".format(
        df.shape[1], num_columns, nan_threshold
    ))
    return df


def _fill_nan(
    df: pd.DataFrame,
    categorical_fill: object = "Missing",
    numerical_fill: object = -1
) -> pd.DataFrame:
    """
    Fills nan values in the dataset with provided rules.
    """
    df = df.copy()
    for col in df.columns:
        if col in CATEGORICAL_TRANS:
            # Categorical columns.
            df[col].fillna(categorical_fill, inplace=True)
        else:
            # Numerical columns.
            # df[col].fillna(numerical_fill(df[col]), inplace=True)
            df[col].fillna(-1, inplace=True)
    assert not np.any(df.isna())
    return df


def PCA_reduction(
    df: pd.DataFrame,
    cols: List[str],
    n_components: int,
    prefix: str = 'PCA_',
    random_seed: int = 42,
    keep: bool = False
) -> pd.DataFrame:
    """
    Substitutes given feature columns with their principal components.
    Args:
        df:
            Dataframe to work on.
        cols:
            Feature columns to apply PCA.
        n_components:
            Number of principal components to be generated.
        prefix:
            Name of principal component feature columns.
        keep:
            Whether to keep raw feature columns, use True only for debugging purpose.
    Returns:
        Modified copy of df.
    """
    df = df.copy()
    pca = decomposition.PCA(n_components=n_components, random_state=random_seed)

    principal_components = pca.fit_transform(df[cols])

    principal_df = pd.DataFrame(principal_components)
    if not keep:
        df.drop(cols, axis=1, inplace=True)

    principal_df.rename(columns=lambda x: str(prefix) + str(x), inplace=True)

    # Align index of principal components and the original dataset.
    principal_df = principal_df.set_index(df.index)

    df = pd.concat([df, principal_df], axis=1)

    return df


def pca_and_cluster(
    df: pd.DataFrame,
    columns: str,
    n_components: Union[int, None],
    n_clusters: Union[int, None],
    keep: bool = False
) -> pd.DataFrame:
    """
    Dimension reduction.
    This method firstly run PCA on selected feature columns and then run k-means.
    Outcome:
        Original df size: (n, k)
        Output df size: (n, `n_components` + 1) where the extra feature column is the cluster index from kmean.
    """
    col_id = columns.upper()
    print("Executing dimension reduction (pca + kmean) on {}* features...".format(col_id))
    selected_columns = [x for x in df.columns if x.startswith(col_id)]
    print("Number of corresponding columns before processing: {}".format(len(selected_columns)))
    for col in selected_columns:
        # Fill Nans with -2. Tentative.
        df[col].fillna((df[col].min() - 2), inplace=True)
        df[col] = preprocessing.minmax_scale(df[col], feature_range=(0, 1))
    df = PCA_reduction(df, selected_columns, prefix="PCA_{}_".format(col_id), n_components=n_components, keep=keep)

    # Apply kmeans on PCA columns
    s = "PCA_{}_".format(col_id)
    pca_columns = [x for x in df.columns if x.startswith(s)]
    kmean = cluster.KMeans(n_clusters=n_clusters)
    kmean_fit = kmean.fit(df[pca_columns])
    df["cluster_{}".format(col_id)] = kmean_fit.predict(df[pca_columns])
    # predicted_indices = kmean_fit.predict(df[pca_columns])
    # df["cluster_{}".format(col_id)] = kmean_fit.cluster_centers_[predicted_indices]
    return df


def _prune_categories(
    df: pd.DataFrame,
    prune_dict: dict,
    fill: object = "Other",
) -> pd.DataFrame:
    """
    Caps some categories variables.
    """
    df = df.copy()
    for col in df.columns:
        if col in CATEGORICAL_TRANS:
            if col not in prune_dict.keys():
                continue
            n = prune_dict[col]
            if n == -1:
                continue
            # Get most frequent:
            major_categories = list(
                df[col].value_counts()[:n].keys()
            )
            mask = df[col].isin(major_categories)
            df[col][~ mask] = fill
    return df


def _convert_to_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts categorical variables to dummies.
    """
    df = df.copy()
    print("Raw dataset shape = {}".format(df.shape))
    print("Creating one hot encoded categories...")
    for col in df.columns:
        if col in CATEGORICAL_TRANS and col != "isFraud":
            # Convert to categorical type, may not be necessary.
            df[col] = pd.Categorical(df[col])
            one_hot_encoded = pd.get_dummies(df[col], prefix=col)
            df = df.drop(columns=[col])  # remove the original categorical column.
            # Add the one-hot-encoded column.
            df = pd.concat([df, one_hot_encoded], axis=1)
    print("One-hot-encoded dataset shape = {}".format(df.shape))
    return df


def report_unique_values(df) -> None:
    """
    Just a helper function.
    """
    report = "{}: {} unique values."
    for col in CATEGORICAL_TRANS:
        try:
            print(report.format(col, len(set(df[col]))))
        except:
            pass


if __name__ == "__main__":
    pass
