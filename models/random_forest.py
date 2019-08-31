"""
Random forest.
"""
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from models import generic

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
    X_train: Union[pd.DataFrame, np.ndarray] = None,
    y_train: Union[pd.DataFrame, np.ndarray] = None,
    X_test: Union[pd.DataFrame, np.ndarray] = None,
    prediction_path: Optional[str] = None,
    estimate_error: bool = False
) -> Optional[pd.ndarray]:
    r = generic.predict(
        build_model=RandomForestClassifier,
        params=PARAMS,
        score=SCORE,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        prediction_path=prediction_path,
        estimate_error=estimate_error
    )
    if prediction_path is not None:
        return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", type=str, default=None
    )
    parser.add_argument(
        "--logdir", type=str, default=None
    )
    args = parser.parse_args()

    # TODO: add X_train, y_..., X_... = utils.get_data()...
    if args.task.lower() in ["predict", "p"]:
        print("Execute: {}".format(args.task))
        if args.logdir is None:
            print("No log directory is provided, no submission file will be generated.")
        predict(...)
    elif args.task.lower() in ["search", "s"]:
        print("Execute: {}".format(args.task))
        print("Execute task: {}".format(args.task))
        if args.logdir is None:
            print("No log directory is provided, best model chosen will only be printed.")
        grid_search(...)
    else:
        raise SyntaxError(
            "Task provided not avaiable: {}".format(args.task))
