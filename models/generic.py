"""
Methods for a generic type of model.
"""
from typing import Union, Optional, Callable, Dict
from pprint import pprint

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from utils import data_utils


def predict(
    build_model: Callable,
    params: Dict[str, object],
    X_train: Union[pd.DataFrame, np.ndarray] = None,
    y_train: Union[pd.DataFrame, np.ndarray] = None,
    X_test: Union[pd.DataFrame, np.ndarray] = None,
    prediction_path: Optional[str] = None,
    estimate_error: bool = False
) -> Optional[np.ndarray]:
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
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError
    print("Datasets:")
    s = "X_train@{}, y_train@{}, X_test@{}"
    print(s.format(X_train.shape, y_train.shape, X_test.shape))
    print("Model Parameters:")
    pprint(params)
    # Save raw data:
    raw_X_train = X_train.copy()
    raw_y_train = y_train.copy()
    raw_X_test = X_test.copy()
    # Error estimation phase:
    print("Phase 1: Estimate Performance by 50%-50% CV...")
    if estimate_error:
        X_train, X_dev, y_train, y_dev = train_test_split(
            X_train, y_train, test_size=0.5, random_state=42
        )
        model = build_model(
            **params,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train.values, y_train.values)
        pred_dev = model.predict_proba(X_dev)[:, 1]
        report_preformance(true=y_dev, prob=pred_dev)
        del model
    else:
        print("Skipped.")

    print("Phase 2")
    model = build_model(
        **params,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    # Retrive the raw datasets, may be unnecessary.
    X_train = raw_X_train.copy()
    y_train = raw_y_train.copy()
    X_test = raw_X_test.copy()

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
        return None


def report_preformance(true, prob) -> None:
    log_loss = metrics.log_loss(y_true=true, y_pred=prob)
    auc = metrics.roc_auc_score(y_true=true, y_pred=prob)
    s = "Log Loss: {:0.6f}, AUC ROC: {:0.6f}"
    print(s.format(log_loss, auc))
