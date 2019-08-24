# IEEE CIS Fraud Detection Competition

IEEE-CIS Fraud Detection Competition on Kaggle

### Notes on Directory
#### Utilities

#### Models

#### Datasets

Use `./data/` to store training and test datasets. Files under `./data/` are added to `./.gitignore`.

#### Naming Conventions

1. Directories are named `path` or `file_dir` and, by default, the path strings do *not* end with `/`.

2. Use upper-case `X` to denote the design matrix (i.e. feature matrix), both data frames and numpy arrays.

3. Use lower-case `y` to denote the label matrix, both data frames and numpy arrays.
4. Data sets are split into `df_train, df_dev, df_test`, where `df_dev` serves as the validation set.

#### Notes on Workflow

1. Use `numpy` and `pandas` for preliminary data preprocessing, including
   1. Dropping invalid observations;
   2. Dropping inferior columns/features (e.g. those with too many invalid observations).
   3. etc.
2. Use `tf.data` and `tf.feature_column` modules to manage dataset, and bridge preprocessed datasets and models ([feature columns](https://www.tensorflow.org/guide/feature_columns))
3. Use `tf.estimators` as baseline models.
4. [*Optional*] Use `tf.keras` and other lower level APIs for model customizations.

#### Style Guide

[Google Python Styleguide](http://google.github.io/styleguide/pyguide.html)