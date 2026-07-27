"""Unit tests for src.preprocessing."""

import pandas as pd

from src import config
from src.preprocessing import build_preprocessing_pipeline, encode_target, split_features_target


def _sample_df() -> pd.DataFrame:
    data = {
        "age": [39, 50],
        "workclass": ["State-gov", "Self-emp-not-inc"],
        "fnlwgt": [77516, 83311],
        "education": ["Bachelors", "Bachelors"],
        "education-num": [13, 13],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [2174, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 13],
        "native-country": ["United-States", "United-States"],
        "income": ["<=50K", "<=50K"],
    }
    return pd.DataFrame(data)


def test_encode_target():
    df = _sample_df()
    df.loc[1, "income"] = ">50K"
    y = encode_target(df)
    assert list(y) == [0, 1]


def test_split_features_target():
    df = _sample_df()
    X, y = split_features_target(df)
    expected_cols = set(config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)
    assert set(X.columns) == expected_cols
    assert len(y) == len(df)


def test_preprocessing_pipeline_fits_and_transforms():
    df = _sample_df()
    X, _ = split_features_target(df)
    pipeline = build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(X)
    assert transformed.shape[0] == len(df)
