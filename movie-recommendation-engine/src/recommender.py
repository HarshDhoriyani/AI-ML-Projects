"""
recommender.py
--------------
Content-based (genre-driven) movie recommendation engine.

Core idea
---------
1. Every movie is represented as a multi-hot vector over the genre space
   (e.g. Action=1, Comedy=0, Sci-Fi=1, ...).
2. A "user profile" is built as a weighted average of the genre vectors of
   movies the user has already rated -- movies rated higher contribute more
   to the profile. Alternatively, a user profile can be built directly from
   a list of preferred genres (useful for new/cold-start users).
3. Recommendations are produced by ranking *unseen* movies according to the
   cosine similarity between their genre vector and the user profile vector,
   optionally re-weighted by the movie's average community rating.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class GenreRecommender:
    """Genre-based content recommender."""

    def __init__(self, movies: pd.DataFrame, ratings: Optional[pd.DataFrame] = None):
        """
        Parameters
        ----------
        movies : DataFrame with columns [movieId, title, genres, ...]
        ratings : optional DataFrame with columns [userId, movieId, rating]
                  used for (a) building profiles from history and
                  (b) computing popularity/average-rating boosts.
        """
        self.movies = movies.reset_index(drop=True).copy()
        self.ratings = ratings

        # Multi-hot genre matrix: rows = movies, columns = genres
        genre_dummies = self.movies["genres"].str.get_dummies(sep="|")
        self.genre_columns = list(genre_dummies.columns)
        self.genre_matrix = genre_dummies.values.astype(float)

        # Pre-compute average rating per movie (used as a popularity tiebreaker)
        if self.ratings is not None:
            avg = self.ratings.groupby("movieId")["rating"].mean()
            self.movies["avg_rating"] = self.movies["movieId"].map(avg)
        else:
            self.movies["avg_rating"] = np.nan

    # ------------------------------------------------------------------ #
    # Profile construction
    # ------------------------------------------------------------------ #
    def build_profile_from_ratings(self, user_id: int) -> np.ndarray:
        """
        Build a user genre-preference vector from their rating history.
        Movies rated higher contribute proportionally more weight.
        """
        if self.ratings is None:
            raise ValueError("No ratings data was provided to the recommender.")

        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        if user_ratings.empty:
            raise ValueError(f"No ratings found for userId={user_id}")

        merged = user_ratings.merge(self.movies, on="movieId", how="left")

        weighted_sum = np.zeros(len(self.genre_columns))
        weight_total = 0.0

        for _, row in merged.iterrows():
            idx = self.movies.index[self.movies["movieId"] == row["movieId"]][0]
            genre_vec = self.genre_matrix[idx]
            weight = float(row["rating"])
            weighted_sum += genre_vec * weight
            weight_total += weight

        if weight_total == 0:
            raise ValueError("Could not build a profile: total rating weight is zero.")

        return weighted_sum / weight_total

    def build_profile_from_genres(self, preferred_genres: Iterable[str]) -> np.ndarray:
        """
        Build a user genre-preference vector directly from a list of
        preferred genres (equal weight each). Useful for new users with
        no rating history (cold start).
        """
        profile = np.zeros(len(self.genre_columns))
        preferred_genres = [g.strip() for g in preferred_genres]

        matched = 0
        for genre in preferred_genres:
            if genre in self.genre_columns:
                profile[self.genre_columns.index(genre)] = 1.0
                matched += 1

        if matched == 0:
            raise ValueError(
                f"None of the requested genres {preferred_genres} were found. "
                f"Available genres: {self.genre_columns}"
            )
        return profile

    # ------------------------------------------------------------------ #
    # Recommendation
    # ------------------------------------------------------------------ #
    def recommend(
        self,
        user_profile: np.ndarray,
        n: int = 5,
        exclude_movie_ids: Optional[Iterable[int]] = None,
        popularity_weight: float = 0.15,
    ) -> pd.DataFrame:
        """
        Return the top-n movies matching the given user profile.

        Parameters
        ----------
        user_profile : genre preference vector (from build_profile_from_*)
        n : number of recommendations to return
        exclude_movie_ids : movie IDs to exclude (e.g. already watched)
        popularity_weight : how much the average community rating
                             (0-5 scale, normalized to 0-1) should influence
                             the final ranking on top of genre similarity.
                             0 = pure content-based, 1 = pure popularity.

        Returns
        -------
        DataFrame with columns [movieId, title, genres, similarity, avg_rating, score]
        sorted by score descending.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")

        candidates = self.movies.copy()
        if exclude_movie_ids:
            candidates = candidates[~candidates["movieId"].isin(exclude_movie_ids)]

        if candidates.empty:
            return candidates.assign(similarity=[], score=[])

        candidate_indices = candidates.index.to_numpy()
        candidate_genre_matrix = self.genre_matrix[candidate_indices]

        similarities = cosine_similarity(
            user_profile.reshape(1, -1), candidate_genre_matrix
        ).flatten()

        candidates = candidates.assign(similarity=similarities)

        # Normalize average rating (0-5 scale) to 0-1 for blending; missing -> 0
        norm_rating = candidates["avg_rating"].fillna(0) / 5.0

        candidates["score"] = (
            (1 - popularity_weight) * candidates["similarity"]
            + popularity_weight * norm_rating
        )

        top_n = candidates.sort_values("score", ascending=False).head(n)

        display_cols = ["movieId", "title", "genres", "similarity", "avg_rating", "score"]
        return top_n[display_cols].reset_index(drop=True)
