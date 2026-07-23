# 🎬 Movie Recommendation Engine

A lightweight **content-based (genre-driven) movie recommendation system**.
Given a user's genre preferences — either inferred from their past ratings
or specified directly — the engine returns the **top-N** movies that best
match their taste.

> Input: `n = 5` → Output: top 5 movies recommended according to the user's genre profile.

---

## How it works

1. **Genre encoding** — Every movie is converted into a multi-hot vector over
   the full genre space (e.g. `Action=1, Comedy=0, Sci-Fi=1, ...`) using its
   pipe-separated `genres` field.
2. **User profile** — A profile vector is built in one of two ways:
   - **From rating history**: a weighted average of the genre vectors of
     movies the user has already rated (higher ratings → more influence).
   - **From stated preferences**: a one-hot vector built directly from a
     list of genres the user says they like (great for cold-start users
     with no history).
3. **Scoring & ranking** — Cosine similarity is computed between the user
   profile and every *unseen* movie's genre vector. This is optionally
   blended with the movie's average community rating (a small popularity
   boost) to break ties and avoid recommending obscure, poorly-rated films.
4. **Top-N output** — The N highest-scoring movies are returned.

---

## Project structure

```
movie-recommendation-engine/
├── app.py                     # CLI entry point
├── data/
│   ├── movies.csv              # movieId, title, year, genres
│   └── ratings.csv             # userId, movieId, rating
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # CSV loading & validation helpers
│   └── recommender.py          # GenreRecommender class (core ML logic)
├── tests/
│   ├── __init__.py
│   └── test_recommender.py     # pytest unit tests
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/HarshDhoriyani/Movie-Recommendation-Engine.git
cd movie-recommendation-engine
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### 1. Recommend based on an existing user's rating history

```bash
python app.py --user-id 1 --n 5
```

```
Top 5 recommendations based on user #1's rating history:

1. Mad Max: Fury Road  [Action|Adventure|Sci-Fi]  (match: 0.87, avg rating: 4.0, score: 0.86)
2. The Avengers        [Action|Adventure|Sci-Fi]  (match: 0.87, avg rating: N/A, score: 0.74)
3. Die Hard            [Action|Thriller]           (match: 0.82, avg rating: N/A, score: 0.70)
4. Jurassic Park       [Adventure|Sci-Fi|Thriller] (match: 0.71, avg rating: N/A, score: 0.60)
5. Gladiator           [Action|Adventure|Drama]     (match: 0.58, avg rating: N/A, score: 0.49)
```

### 2. Recommend based on stated genre preferences (cold start)

```bash
python app.py --genres "Animation,Family,Comedy" --n 5
```

### 3. List all available genres in the dataset

```bash
python app.py --list-genres
```

### 4. Use your own dataset

```bash
python app.py --user-id 3 --n 5 \
    --movies-path path/to/movies.csv \
    --ratings-path path/to/ratings.csv
```

Any MovieLens-style dataset works out of the box as long as it has:
- `movies.csv` → `movieId, title, genres` (genres pipe-separated: `Action|Comedy`)
- `ratings.csv` → `userId, movieId, rating`

---

## Using it as a library

```python
from src.data_loader import load_movies, load_ratings
from src.recommender import GenreRecommender

movies = load_movies("data/movies.csv")
ratings = load_ratings("data/ratings.csv")

recommender = GenreRecommender(movies, ratings)

# Build a profile from a user's history
profile = recommender.build_profile_from_ratings(user_id=1)

# Or build one from stated preferences directly
# profile = recommender.build_profile_from_genres(["Action", "Sci-Fi"])

top_5 = recommender.recommend(profile, n=5)
print(top_5)
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Possible extensions

- Swap in a **TF-IDF weighted genre/tag representation** for finer-grained similarity.
- Add **collaborative filtering** (user-user or item-item) and blend it with the
  content-based score for a hybrid recommender.
- Wrap the engine in a **Flask/FastAPI** service for a web-based demo.
- Plug in the full **MovieLens 25M dataset** for large-scale experimentation.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
