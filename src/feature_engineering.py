"""
feature_engineering.py
-----------------------
Builds per-user behavioral features from raw interaction logs.

Feature philosophy:
    - Completion rate     → did the user finish what they started? (quality signal)
    - Interaction intensity → how much did they consume overall? (volume signal)
    - Content diversity   → did they explore broadly or go deep on one genre?
    - Temporal consistency → did they watch steadily or in bursts? (loyalty signal)
    - Scoring tendency    → how critical vs generous is this user?

These mirror the features a streaming platform like Netflix would compute
to understand member engagement depth and churn risk.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# MyAnimeList status codes
STATUS_MAP = {
    1: "watching",
    2: "completed",
    3: "on_hold",
    4: "dropped",
    6: "plan_to_watch",
}


def build_user_features(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate interaction-level data into one row per user.

    Parameters
    ----------
    interactions : pd.DataFrame
        Must contain: user_id, anime_id, my_score, my_watched_episodes,
                      my_status, my_times_watched

    Returns
    -------
    pd.DataFrame : one row per user_id with engineered features
    """
    logger.info("Building user-level behavioral features...")
    df = interactions.copy()

    # ── Status flags ──────────────────────────────────────────────
    df["is_completed"] = (df["my_status"] == 2).astype(int)
    df["is_dropped"]   = (df["my_status"] == 4).astype(int)
    df["is_watching"]  = (df["my_status"] == 1).astype(int)
    df["has_score"]    = (df["my_score"] > 0).astype(int)
    df["rewatched"]    = (df["my_times_watched"] > 1).astype(int)

    # ── Per-user aggregations ─────────────────────────────────────
    agg = df.groupby("user_id").agg(
        total_interactions      = ("anime_id",            "count"),
        total_completed         = ("is_completed",        "sum"),
        total_dropped           = ("is_dropped",          "sum"),
        total_watching          = ("is_watching",         "sum"),
        total_scored            = ("has_score",           "sum"),
        total_rewatched         = ("rewatched",           "sum"),
        mean_score              = ("my_score",            lambda x: x[x > 0].mean()),
        score_std               = ("my_score",            lambda x: x[x > 0].std()),
        total_episodes_watched  = ("my_watched_episodes", "sum"),
        unique_titles           = ("anime_id",            "nunique"),
    ).reset_index()

    # ── Derived ratios ────────────────────────────────────────────

    # Completion rate: fraction of started titles that were finished
    # (drops plan_to_watch from denominator)
    started = agg["total_interactions"] - df.groupby("user_id")["my_status"].apply(
        lambda x: (x == 6).sum()
    ).reset_index(drop=True)
    agg["completion_rate"] = (agg["total_completed"] / started.clip(lower=1)).clip(0, 1)

    # Drop rate
    agg["drop_rate"] = (agg["total_dropped"] / agg["total_interactions"].clip(lower=1)).clip(0, 1)

    # Scoring rate: fraction of interactions that received an explicit score
    agg["scoring_rate"] = (agg["total_scored"] / agg["total_interactions"].clip(lower=1)).clip(0, 1)

    # Rewatch rate: signals deep engagement with content
    agg["rewatch_rate"] = (agg["total_rewatched"] / agg["total_interactions"].clip(lower=1)).clip(0, 1)

    # Interaction intensity: log-scale total episodes watched (heavy users penalised less)
    agg["log_episodes_watched"] = np.log1p(agg["total_episodes_watched"])

    # Score consistency: low std = consistent taste, high std = eclectic
    agg["score_std"] = agg["score_std"].fillna(0)

    # ── Engagement tier label (used later for retention modeling) ─
    # High engagement = completed many, rarely drops, actively scores
    engagement_score = (
        agg["completion_rate"] * 0.5
        + agg["scoring_rate"]   * 0.3
        + agg["rewatch_rate"]   * 0.2
    )
    agg["high_engagement"] = (engagement_score >= engagement_score.quantile(0.65)).astype(int)

    logger.info(f"Feature matrix shape: {agg.shape}")
    logger.info(f"High-engagement users: {agg['high_engagement'].mean():.1%} of total")
    return agg


def build_cohort_features(interactions: pd.DataFrame,
                           user_meta: pd.DataFrame = None) -> pd.DataFrame:
    """
    Assign users to onboarding vs long-term cohorts based on
    interaction volume — a lightweight proxy for tenure when
    join_date is unavailable.

    Cohorts:
        'new'      : bottom 33% by interaction count
        'growing'  : middle 33%
        'retained' : top 33%
    """
    user_counts = interactions.groupby("user_id")["anime_id"].count().reset_index()
    user_counts.columns = ["user_id", "interaction_count"]

    q33 = user_counts["interaction_count"].quantile(0.33)
    q66 = user_counts["interaction_count"].quantile(0.66)

    def assign_cohort(n):
        if n <= q33:
            return "new"
        elif n <= q66:
            return "growing"
        return "retained"

    user_counts["cohort"] = user_counts["interaction_count"].apply(assign_cohort)
    logger.info(f"Cohort distribution:\n{user_counts['cohort'].value_counts()}")
    return user_counts


if __name__ == "__main__":
    from data_loader import load_interactions
    interactions = load_interactions(sample_users=10_000)
    features = build_user_features(interactions)
    cohorts  = build_cohort_features(interactions)
    print(features.describe())
    print(cohorts["cohort"].value_counts())
