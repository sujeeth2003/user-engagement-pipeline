"""
run_pipeline.py
---------------
End-to-end pipeline runner.

Usage:
    python run_pipeline.py --users 70000

Steps:
    1. Load & validate raw interaction data
    2. Engineer per-user behavioral features
    3. Build cohort labels and run cohort analysis
    4. Cluster users into segments (K-Means)
    5. Train and evaluate retention model (LightGBM)
    6. Save all outputs to outputs/
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader        import load_interactions, load_anime_metadata, validate_interactions
from feature_engineering import build_user_features, build_cohort_features
from cohort_analysis    import merge_cohorts, cohort_summary, plot_cohort_engagement, plot_engagement_distributions
from clustering         import run_clustering, find_optimal_k, select_features
from retention_model    import run_retention_pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/pipeline.log"),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="User Engagement Analytics Pipeline")
    parser.add_argument("--users",    type=int, default=70_000, help="Number of users to sample")
    parser.add_argument("--clusters", type=int, default=4,      help="Number of K-Means clusters")
    parser.add_argument("--find-k",   action="store_true",      help="Run elbow/silhouette to find optimal K")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("outputs", exist_ok=True)

    # ── Step 1: Load data ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)
    interactions = load_interactions(sample_users=args.users)
    validate_interactions(interactions)
    anime = load_anime_metadata()
    logger.info(f"Loaded {interactions['user_id'].nunique():,} users × {interactions['anime_id'].nunique():,} items")

    # ── Step 2: Feature engineering ────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Engineering behavioral features")
    logger.info("=" * 60)
    features = build_user_features(interactions)

    # ── Step 3: Cohort analysis ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Cohort analysis")
    logger.info("=" * 60)
    cohorts = build_cohort_features(interactions)
    df_cohort = merge_cohorts(features, cohorts)
    summary = cohort_summary(df_cohort)
    plot_cohort_engagement(summary)
    plot_engagement_distributions(df_cohort)

    # ── Step 4: User segmentation ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: K-Means user segmentation")
    logger.info("=" * 60)

    if args.find_k:
        from sklearn.preprocessing import StandardScaler
        X = select_features(features).fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        k = find_optimal_k(X_scaled)
    else:
        k = args.clusters

    features_clustered, scaler, km = run_clustering(features, k=k)
    logger.info(f"Segment distribution:\n{features_clustered['cluster_label'].value_counts()}")

    # ── Step 5: Retention modeling ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Retention modeling")
    logger.info("=" * 60)
    results = run_retention_pipeline(features)
    logger.info(f"Final Test ROC-AUC: {results['roc_auc']:.4f}")

    # ── Summary ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Users processed : {features['user_id'].nunique():,}")
    logger.info(f"  Items in dataset: {interactions['anime_id'].nunique():,}")
    logger.info(f"  Clusters        : {k}")
    logger.info(f"  Retention AUC   : {results['roc_auc']:.4f}")
    logger.info(f"  Outputs saved to: outputs/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
