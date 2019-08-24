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


def clean_dat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the original dataset and formulates it so that
    it is compatible with most existing ML models.
    """

