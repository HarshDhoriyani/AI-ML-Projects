"""
app.py
------
Command-line interface for the Movie Recommendation Engine.

Usage examples
--------------
# Recommend based on an existing user's rating history:
    python app.py --user-id 1 --n 5

# Recommend based on preferred genres directly (cold start):
    python app.py --genres "Action,Sci-Fi" --n 5

# List all available genres:
    python app.py --list-genres
"""

import argparse
import sys

from src.data_loader import load_movies, load_ratings, get_all_genres
from src.recommender import GenreRecommender


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genre-based Movie Recommendation Engine"
    )
    parser.add_argument(
        "--user-id", type=int, default=None,
        help="Build recommendations from this user's rating history."
    )
    parser.add_argument(
        "--genres", type=str, default=None,
        help='Comma-separated preferred genres, e.g. "Action,Comedy"'
    )
    parser.add_argument(
        "--n", type=int, default=5,
        help="Number of movies to recommend (default: 5)."
    )
    parser.add_argument(
        "--popularity-weight", type=float, default=0.15,
        help="Blend factor between genre similarity and average rating (0-1)."
    )
    parser.add_argument(
        "--movies-path", type=str, default="data/movies.csv",
        help="Path to movies CSV file."
    )
    parser.add_argument(
        "--ratings-path", type=str, default="data/ratings.csv",
        help="Path to ratings CSV file."
    )
    parser.add_argument(
        "--list-genres", action="store_true",
        help="Print all available genres and exit."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    movies = load_movies(args.movies_path)

    if args.list_genres:
        print("Available genres:")
        for g in get_all_genres(movies):
            print(f"  - {g}")
        sys.exit(0)

    ratings = None
    try:
        ratings = load_ratings(args.ratings_path)
    except FileNotFoundError:
        ratings = None

    recommender = GenreRecommender(movies, ratings)

    watched_ids = []
    if args.user_id is not None:
        if ratings is None:
            print("Error: ratings file is required to build a profile from user history.")
            sys.exit(1)
        profile = recommender.build_profile_from_ratings(args.user_id)
        watched_ids = ratings.loc[
            ratings["userId"] == args.user_id, "movieId"
        ].tolist()
        source = f"user #{args.user_id}'s rating history"
    elif args.genres:
        genre_list = args.genres.split(",")
        profile = recommender.build_profile_from_genres(genre_list)
        source = f"preferred genres: {', '.join(g.strip() for g in genre_list)}"
    else:
        print("Error: provide either --user-id or --genres (or --list-genres).")
        sys.exit(1)

    results = recommender.recommend(
        user_profile=profile,
        n=args.n,
        exclude_movie_ids=watched_ids,
        popularity_weight=args.popularity_weight,
    )

    print(f"\nTop {args.n} recommendations based on {source}:\n")
    if results.empty:
        print("No recommendations found.")
        return

    for rank, row in enumerate(results.itertuples(), start=1):
        avg_rating = f"{row.avg_rating:.1f}" if row.avg_rating == row.avg_rating else "N/A"
        print(
            f"{rank}. {row.title}  "
            f"[{row.genres}]  "
            f"(match: {row.similarity:.2f}, avg rating: {avg_rating}, score: {row.score:.2f})"
        )


if __name__ == "__main__":
    main()
