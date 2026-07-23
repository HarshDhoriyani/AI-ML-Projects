"""
data_loader.py
--------------
Utility functions to load and validate the movies and ratings datasets.
"""

import os
import pandas as pd

REQUIRED_MOVIE_COLUMNS = {"movieId", "title", "genres"}
REQUIRED_RATING_COLUMNS = {"userId", "movieId", "rating"}


def load_movies(path: str = "data/movies.csv") -> pd.DataFrame:
    """
    Load the movies dataset.

    Expected columns: movieId, title, year, genres
    Genres are pipe-separated, e.g. "Action|Sci-Fi|Thriller"
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Movies file not found at: {path}")

    movies = pd.read_csv(path)

    missing = REQUIRED_MOVIE_COLUMNS - set(movies.columns)
    if missing:
        raise ValueError(f"movies.csv is missing required columns: {missing}")

    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    return movies


def load_ratings(path: str = "data/ratings.csv") -> pd.DataFrame:
    """
    Load the ratings dataset.

    Expected columns: userId, movieId, rating
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ratings file not found at: {path}")

    ratings = pd.read_csv(path)

    missing = REQUIRED_RATING_COLUMNS - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings.csv is missing required columns: {missing}")

    return ratings


def get_all_genres(movies: pd.DataFrame) -> list:
    """Return a sorted list of all unique genres present in the dataset."""
    genre_set = set()
    for genre_string in movies["genres"]:
        genre_set.update(genre_string.split("|"))
    return sorted(genre_set)


def get_user_rated_movies(ratings: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """Return all ratings given by a specific user."""
    user_ratings = ratings[ratings["userId"] == user_id]
    if user_ratings.empty:
        raise ValueError(f"No ratings found for userId={user_id}")
    return user_ratings
