"""
Methods for a generic type of model.
"""
from typing import Union, Optional, Callable

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from utils import data_utils


PARAMS = {'max_features': 'log2', 'criterion': 'gini',
          'n_estimators': 1900, 'max_depth': 64}

# **** add configuration here ****
PARAM_SCOPE = {
    "max_depth": [None] + [2 ** x for x in range(5, 11)],
    "n_estimators": [100 * x for x in range(1, 20, 2)],
    "criterion": ["entropy", "gini"],
    "max_features": ["auto", "sqrt", "log2"],
}

SCORE = "neg_log_loss"


def predict(
    build_model: Callable,
    params: Dict[str, object],
    score: str = "neg_log_loss",
    X_train: Union[pd.DataFrame, np.ndarray] = None,
    y_train: Union[pd.DataFrame, np.ndarray] = None,
    X_test: Union[pd.DataFrame, np.ndarray] = None,
    prediction_path: Optional[str] = None,
    estimate_error: bool = False
) -> Optional[pd.ndarray]:
    """
    Generates the classification result for the given dataset.
    If a path of destination is given, this methods will write
    classification prediction to the destination file. Otherwise,
    the prediction on test set is returned.
    Args:
        build_model:
            A method to build the model, like RandomForestClassifier.
        params:
            Parameter dictionary.
        score:
            A measure for the performance of model.
        X_train, y_train, X_test:
            Datasets.
        prediction_path:
            The path used to store prediction made, should end
            with *.csv.
        estimate_error:
            If error estimation is required, the model will firstly
            draw 50%-50% split on the training set (train vs dev)
            and estimate the performance on the dev set. Then,
            the model is fit again on the whole training set, and
            predicted propensities of test set are computed.
    """
    # Check and report datasets:
    # TODO: implement.
    # Error estimation phase:
    print("Phase 1: Estimate Performance by 50%-50% CV...")
    if estimate_error:
        raise NotImplementedError
        # TODO: implement this.
    else:
        print("Skipped.")

    print("Phase 2")
    del model
    model = build_model(
        **params,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train.values, y_train.values)
    print("Predicting on the test set ...")
    pred_test = model.predict_proba(X_test.values)

    # Extract the probability for class == 1.
    pred_test = pred_test[:, 1]
    if prediction_path is None:
        return pred_test
    else:
        print("Write submission file to {}".format(prediction_path))
        data_utils.generate_submission(
            prob=pred_test,
            dest_path=prediction_path,
            src_path="./data"
        )
