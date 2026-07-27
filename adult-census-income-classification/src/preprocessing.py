"""Feature preprocessing pipeline for the Adult Census Income dataset."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build a ColumnTransformer that scales numeric features and
    one-hot encodes categorical features.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def encode_target(df: pd.DataFrame) -> pd.Series:
    """Convert the target column into a binary 0/1 label.

    1 -> income > 50K, 0 -> income <= 50K
    """
    return (df[config.TARGET_COLUMN].str.strip() == ">50K").astype(int)


def split_features_target(df: pd.DataFrame):
    """Split a raw DataFrame into (X, y)."""
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES].copy()
    y = encode_target(df)
    return X, y
