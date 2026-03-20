"""
clustering.py
-------------
Segments users into interpretable behavioral clusters using K-Means.

Goal: surface distinct user personas from the feature space —
e.g., "completionist power users", "casual browsers", "selective scorers".
These segments feed into downstream retention modeling and can drive
personalisation decisions (what content to surface, when to nudge).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
import logging

logger = logging.getLogger(__name__)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


FEATURE_COLS = [
    "completion_rate",
    "drop_rate",
    "scoring_rate",
    "rewatch_rate",
    "log_episodes_watched",
    "score_std",
    "mean_score",
]

CLUSTER_NAMES = {
    # These are assigned after inspecting cluster centroids
    # Updated in label_clusters() based on actual data
}


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and clean the feature matrix for clustering."""
    X = df[FEATURE_COLS].copy()
    X = X.fillna(X.median())
    return X


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> int:
    """
    Elbow + silhouette method to select K.
    Plots inertia and silhouette scores; returns recommended K.
    """
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels, sample_size=5000))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, marker="o", color="#2171B5")
    axes[0].set_title("Elbow Method — Inertia vs K", fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), silhouettes, marker="o", color="#08306B")
    axes[1].set_title("Silhouette Score vs K", fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_k_selection.png"), dpi=150, bbox_inches="tight")
    plt.show()

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    logger.info(f"Recommended K by silhouette: {best_k}")
    return best_k


def fit_kmeans(X_scaled: np.ndarray, k: int = 4) -> KMeans:
    """Fit K-Means with the chosen K."""
    km = KMeans(n_clusters=k, random_state=42, n_init=15, max_iter=300)
    km.fit(X_scaled)
    logger.info(f"K-Means fitted with K={k}, inertia={km.inertia_:.2f}")
    return km


def label_clusters(df: pd.DataFrame, feature_cols: list = FEATURE_COLS) -> pd.DataFrame:
    """
    After clustering, inspect centroids and assign human-readable labels.
    Labels are derived from the centroid values — completionists have high
    completion_rate, browsers have low scoring_rate, etc.
    """
    centroid_df = df.groupby("cluster")[feature_cols].mean()
    logger.info(f"Cluster centroids:\n{centroid_df.round(3).to_string()}")

    # Heuristic labelling based on dominant centroid characteristics
    labels = {}
    for cluster_id, row in centroid_df.iterrows():
        if row["completion_rate"] > 0.7 and row["log_episodes_watched"] > centroid_df["log_episodes_watched"].median():
            labels[cluster_id] = "Power Completionists"
        elif row["drop_rate"] > 0.3:
            labels[cluster_id] = "Selective Browsers"
        elif row["scoring_rate"] > 0.6 and row["rewatch_rate"] > 0.1:
            labels[cluster_id] = "Engaged Critics"
        else:
            labels[cluster_id] = "Casual Explorers"

    df["cluster_label"] = df["cluster"].map(labels)
    logger.info(f"Cluster label distribution:\n{df['cluster_label'].value_counts()}")
    return df


def plot_clusters_pca(df: pd.DataFrame, X_scaled: np.ndarray, save: bool = True) -> None:
    """
    2D PCA projection of clusters — the most intuitive way to show
    that segments are well-separated.
    """
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        "PC1":    coords[:, 0],
        "PC2":    coords[:, 1],
        "Segment": df["cluster_label"].values,
    })

    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("Blues_d", n_colors=df["cluster_label"].nunique())
    sns.scatterplot(
        data=plot_df, x="PC1", y="PC2", hue="Segment",
        palette=palette, alpha=0.5, s=15, ax=ax, linewidth=0
    )
    ax.set_title("User Segments — PCA Projection", fontweight="bold", fontsize=13)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance explained)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance explained)")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "user_segments_pca.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved PCA cluster plot → {path}")
    plt.show()


def plot_cluster_radar(df: pd.DataFrame, save: bool = True) -> None:
    """
    Radar / spider chart comparing cluster profiles across key features.
    Makes the segment personas immediately interpretable.
    """
    metrics = ["completion_rate", "drop_rate", "scoring_rate",
               "rewatch_rate", "log_episodes_watched"]
    labels  = ["Completion\nRate", "Drop\nRate", "Scoring\nRate",
               "Rewatch\nRate", "Log Episodes\nWatched"]

    centroid_df = df.groupby("cluster_label")[metrics].mean()
    # Normalise to [0,1] for radar readability
    centroid_norm = (centroid_df - centroid_df.min()) / (centroid_df.max() - centroid_df.min() + 1e-9)

    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = ["#08306B", "#2171B5", "#6BAED6", "#BDD7E7"]

    for (segment, row), color in zip(centroid_norm.iterrows(), colors):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, linewidth=2, color=color, label=segment)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("User Segment Profiles", fontweight="bold", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "cluster_radar.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved radar chart → {path}")
    plt.show()


def run_clustering(features: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """
    Full clustering pipeline:
        1. Select + scale features
        2. Fit K-Means
        3. Label clusters
        4. Return enriched DataFrame
    """
    X = select_features(features)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = fit_kmeans(X_scaled, k=k)
    features = features.copy()
    features["cluster"] = km.labels_
    features = label_clusters(features)

    plot_clusters_pca(features, X_scaled)
    plot_cluster_radar(features)

    return features, scaler, km


if __name__ == "__main__":
    from data_loader import load_interactions
    from feature_engineering import build_user_features
    interactions = load_interactions(sample_users=20_000)
    features     = build_user_features(interactions)
    features, scaler, km = run_clustering(features, k=4)
    print(features["cluster_label"].value_counts())
