"""Download and load the Adult Census Income dataset."""

import logging
import urllib.request
from pathlib import Path

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _download(url: str, dest: Path) -> None:
    """Download a file if it doesn't already exist locally."""
    if dest.exists():
        logger.info("Found cached file: %s", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)


def download_dataset() -> None:
    """Download the train and test splits from the UCI repository."""
    _download(config.TRAIN_URL, config.RAW_DATA_PATH)
    _download(config.TEST_URL, config.TEST_DATA_PATH)


def _read_raw(path: Path, skiprows: int = 0) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=config.COLUMN_NAMES,
        na_values=" ?",
        skipinitialspace=True,
        skiprows=skiprows,
    )
    return df


def load_train_data() -> pd.DataFrame:
    """Load the training split as a DataFrame."""
    download_dataset()
    df = _read_raw(config.RAW_DATA_PATH)
    df = df.dropna().reset_index(drop=True)
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].str.strip().str.rstrip(".")
    return df


def load_test_data() -> pd.DataFrame:
    """Load the held-out test split as a DataFrame.

    Note: adult.test has a malformed first line, hence skiprows=1.
    """
    download_dataset()
    df = _read_raw(config.TEST_DATA_PATH, skiprows=1)
    df = df.dropna().reset_index(drop=True)
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].str.strip().str.rstrip(".")
    return df


if __name__ == "__main__":
    train_df = load_train_data()
    test_df = load_test_data()
    logger.info("Train shape: %s | Test shape: %s", train_df.shape, test_df.shape)
    print(train_df.head())
