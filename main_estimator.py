"""
The main script for tensorflow estimators.
"""
import tensorflow as tf

from utils import data_utils
from utils import tf_utils


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = data_utils.load_dataset()
    feature_columns = tf_utils.generate_feature_columns(X_train)

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[32, 32],
        n_classes=2
    )

    classifier.train(
        input_fn=lambda: data_utils.train_input_fn(X_train, y_train, 32),
        steps=10
    )