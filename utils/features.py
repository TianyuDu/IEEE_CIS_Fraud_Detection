"""
Aug. 21, 2019
Use this script to store methods that:
i. Manipulate coulumns of dataset, like creating new features.
"""
import pandas as pd


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
    df = _clean_categorical_transaction(df, fill="missing")
    return df

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


def clean_categorical_transaction(
    df: pd.DataFram
) -> pd.DataFrame:
    """
    Cleans the original dataset and formulates it so that
    it is compatible with most existing ML models.
    """

