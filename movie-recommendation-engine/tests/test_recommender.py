"""
test_recommender.py
--------------------
Unit tests for GenreRecommender. Run with: pytest
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommender import GenreRecommender


@pytest.fixture
def sample_movies():
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5],
        "title": ["Action A", "Comedy B", "Action-Comedy C", "Drama D", "Sci-Fi E"],
        "genres": [
            "Action|Thriller",
            "Comedy",
            "Action|Comedy",
            "Drama",
            "Sci-Fi|Action",
        ],
    })


@pytest.fixture
def sample_ratings():
    return pd.DataFrame({
        "userId": [1, 1, 1, 2],
        "movieId": [1, 5, 4, 2],
        "rating": [5, 4, 2, 5],
    })


def test_genre_matrix_shape(sample_movies):
    rec = GenreRecommender(sample_movies)
    assert rec.genre_matrix.shape[0] == len(sample_movies)
    assert rec.genre_matrix.shape[1] == len(rec.genre_columns)


def test_build_profile_from_genres(sample_movies):
    rec = GenreRecommender(sample_movies)
    profile = rec.build_profile_from_genres(["Action"])
    assert profile[rec.genre_columns.index("Action")] == 1.0


def test_build_profile_from_genres_invalid(sample_movies):
    rec = GenreRecommender(sample_movies)
    with pytest.raises(ValueError):
        rec.build_profile_from_genres(["NotAGenre"])


def test_build_profile_from_ratings(sample_movies, sample_ratings):
    rec = GenreRecommender(sample_movies, sample_ratings)
    profile = rec.build_profile_from_ratings(user_id=1)
    assert isinstance(profile, np.ndarray)
    assert profile.shape[0] == len(rec.genre_columns)
    # user 1 heavily rated Action-themed movies, so Action should dominate
    action_idx = rec.genre_columns.index("Action")
    assert profile[action_idx] > 0


def test_build_profile_from_ratings_missing_user(sample_movies, sample_ratings):
    rec = GenreRecommender(sample_movies, sample_ratings)
    with pytest.raises(ValueError):
        rec.build_profile_from_ratings(user_id=999)


def test_recommend_returns_top_n(sample_movies, sample_ratings):
    rec = GenreRecommender(sample_movies, sample_ratings)
    profile = rec.build_profile_from_genres(["Action"])
    results = rec.recommend(profile, n=3)
    assert len(results) == 3
    assert list(results.columns) == [
        "movieId", "title", "genres", "similarity", "avg_rating", "score"
    ]
    # results should be sorted descending by score
    scores = results["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_recommend_excludes_watched(sample_movies, sample_ratings):
    rec = GenreRecommender(sample_movies, sample_ratings)
    profile = rec.build_profile_from_genres(["Action"])
    results = rec.recommend(profile, n=5, exclude_movie_ids=[1, 3])
    assert 1 not in results["movieId"].values
    assert 3 not in results["movieId"].values


def test_recommend_n_larger_than_available(sample_movies):
    rec = GenreRecommender(sample_movies)
    profile = rec.build_profile_from_genres(["Drama"])
    results = rec.recommend(profile, n=100)
    assert len(results) == len(sample_movies)


def test_recommend_invalid_n(sample_movies):
    rec = GenreRecommender(sample_movies)
    profile = rec.build_profile_from_genres(["Drama"])
    with pytest.raises(ValueError):
        rec.recommend(profile, n=0)
