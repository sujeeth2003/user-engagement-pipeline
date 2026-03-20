"""
cohort_analysis.py
------------------
Computes and visualises engagement metrics broken down by user cohort.

The core idea: users who joined (or are at different lifecycle stages)
behave differently. Surfacing these differences helps answer:
    - Are new users completing content at a healthy rate?
    - Are long-term users rewatching / deeply engaging?
    - Where is the highest drop-off risk?

This is directly analogous to member cohort analysis at a streaming platform —
tracking how engagement evolves from onboarding through retention phases.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COHORT_ORDER   = ["new", "growing", "retained"]
COHORT_PALETTE = {"new": "#6BAED6", "growing": "#2171B5", "retained": "#08306B"}


def merge_cohorts(features: pd.DataFrame, cohorts: pd.DataFrame) -> pd.DataFrame:
    """Join user feature matrix with cohort labels."""
    return features.merge(cohorts[["user_id", "cohort"]], on="user_id", how="left")


def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean engagement metrics per cohort.
    Returns a tidy DataFrame suitable for plotting or reporting.
    """
    metrics = [
        "completion_rate", "drop_rate", "scoring_rate",
        "rewatch_rate", "log_episodes_watched", "mean_score"
    ]
    summary = (
        df.groupby("cohort")[metrics]
        .mean()
        .reindex(COHORT_ORDER)
        .reset_index()
    )
    logger.info(f"Cohort summary:\n{summary.to_string(index=False)}")
    return summary


def plot_cohort_engagement(summary: pd.DataFrame, save: bool = True) -> None:
    """
    Bar chart comparing key engagement metrics across cohorts.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("User Engagement by Lifecycle Cohort", fontsize=14, fontweight="bold", y=1.02)

    metrics_to_plot = [
        ("completion_rate",      "Completion Rate",      "Fraction of started titles completed"),
        ("drop_rate",            "Drop Rate",            "Fraction of titles abandoned"),
        ("log_episodes_watched", "Log Episodes Watched", "Total content consumption (log scale)"),
    ]

    for ax, (metric, title, ylabel) in zip(axes, metrics_to_plot):
        bars = ax.bar(
            summary["cohort"],
            summary[metric],
            color=[COHORT_PALETTE[c] for c in summary["cohort"]],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("User Cohort", fontsize=9)
        ax.set_ylim(0, summary[metric].max() * 1.2)
        ax.tick_params(labelsize=9)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + summary[metric].max() * 0.02,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=9
            )

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "cohort_engagement.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved cohort engagement plot → {path}")
    plt.show()


def plot_engagement_distributions(df: pd.DataFrame, save: bool = True) -> None:
    """
    KDE distributions of completion rate and scoring rate, split by cohort.
    Shows the shape of engagement — not just the mean.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Distribution of Engagement Metrics by Cohort", fontsize=13, fontweight="bold")

    for ax, (col, label) in zip(axes, [
        ("completion_rate", "Completion Rate"),
        ("scoring_rate",    "Scoring Rate"),
    ]):
        for cohort in COHORT_ORDER:
            subset = df[df["cohort"] == cohort][col].dropna()
            sns.kdeplot(subset, ax=ax, label=cohort,
                        color=COHORT_PALETTE[cohort], linewidth=2, fill=True, alpha=0.15)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(title="Cohort")

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "engagement_distributions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved distribution plot → {path}")
    plt.show()


if __name__ == "__main__":
    from data_loader import load_interactions
    from feature_engineering import build_user_features, build_cohort_features

    interactions = load_interactions(sample_users=20_000)
    features     = build_user_features(interactions)
    cohorts      = build_cohort_features(interactions)
    df           = merge_cohorts(features, cohorts)
    summary      = cohort_summary(df)

    plot_cohort_engagement(summary)
    plot_engagement_distributions(df)
