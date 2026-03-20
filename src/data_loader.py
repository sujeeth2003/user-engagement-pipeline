"""
data_loader.py
--------------
Loads and validates the MyAnimeList dataset.

Dataset source: https://www.kaggle.com/datasets/azathoth42/myanimelist
Files needed in data/raw/:
    - UserAnimeList.csv   (user-anime interaction logs)
    - AnimeList.csv       (anime metadata)
    - UserList.csv        (user metadata)
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_interactions(sample_users: int = 70_000) -> pd.DataFrame:
    """
    Load user-anime interaction logs.
    Keeps a random sample of users to keep memory manageable.

    Returns a DataFrame with columns:
        user_id, anime_id, my_score, my_watched_episodes,
        my_status, my_times_watched
    """
    path = os.path.join(RAW_DIR, "UserAnimeList.csv")
    logger.info(f"Loading interactions from {path}")

    cols = [
        "username", "anime_id", "my_score", "my_watched_episodes",
        "my_status", "my_times_watched"
    ]

    df = pd.read_csv(path, usecols=cols, low_memory=False)
    logger.info(f"Raw interactions: {len(df):,} rows")

    # Map usernames to integer IDs
    user_map = {u: i for i, u in enumerate(df["username"].unique())}
    df["user_id"] = df["username"].map(user_map)

    # Sample users for manageability
    if sample_users and df["user_id"].nunique() > sample_users:
        sampled = np.random.choice(df["user_id"].unique(), size=sample_users, replace=False)
        df = df[df["user_id"].isin(sampled)].copy()
        logger.info(f"Sampled down to {df['user_id'].nunique():,} users")

    df = df.drop(columns=["username"])
    logger.info(f"Final interactions shape: {df.shape}")
    return df


def load_anime_metadata() -> pd.DataFrame:
    """
    Load anime metadata (title, genre, type, episodes, members).
    """
    path = os.path.join(RAW_DIR, "AnimeList.csv")
    logger.info(f"Loading anime metadata from {path}")

    cols = ["anime_id", "title", "genre", "type", "episodes", "members", "score"]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
    logger.info(f"Anime metadata: {df.shape[0]:,} titles")
    return df


def load_user_metadata() -> pd.DataFrame:
    """
    Load user-level metadata (join date, days watched).
    """
    path = os.path.join(RAW_DIR, "UserList.csv")
    logger.info(f"Loading user metadata from {path}")

    cols = ["username", "user_days_spent_watching", "user_completed",
            "user_watching", "user_dropped", "join_date"]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df["join_date"] = pd.to_datetime(df["join_date"], errors="coerce")
    logger.info(f"User metadata: {df.shape[0]:,} users")
    return df


def validate_interactions(df: pd.DataFrame) -> None:
    """Basic schema and sanity checks on the interaction table."""
    required = {"user_id", "anime_id", "my_score", "my_watched_episodes", "my_status"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert df["user_id"].nunique() > 1000, "Suspiciously few users"
    assert df["anime_id"].nunique() > 100, "Suspiciously few items"
    assert df["my_score"].between(0, 10).all() or df["my_score"].isnull().any(), \
        "Scores out of expected range"
    logger.info("Validation passed.")


if __name__ == "__main__":
    interactions = load_interactions()
    validate_interactions(interactions)
    anime = load_anime_metadata()
    print(interactions.head())
    print(anime.head())
