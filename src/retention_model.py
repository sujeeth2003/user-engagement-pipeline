"""
retention_model.py
------------------
Trains a Gradient Boosting classifier to predict whether a user will
be a high-engagement (retained) member.

Target variable: high_engagement (1 = top 35% by engagement score)
This is defined in feature_engineering.py and captures users who
complete content, score it, and rewatch — the signals most strongly
correlated with long-term platform loyalty.

Pipeline:
    1. Train / validation / test split (stratified)
    2. Feature scaling
    3. LightGBM classifier
    4. Evaluation: ROC-AUC, classification report, feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report,
    RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
)
import lightgbm as lgb
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
    "total_interactions",
    "unique_titles",
]

TARGET = "high_engagement"


def prepare_data(features: pd.DataFrame):
    """
    Split into train / val / test sets.
    Uses stratified split to preserve class balance.

    Returns X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    df = features.dropna(subset=FEATURE_COLS + [TARGET]).copy()
    X = df[FEATURE_COLS]
    y = df[TARGET]

    logger.info(f"Dataset: {len(df):,} users | Target balance: {y.mean():.1%} high-engagement")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS)
    X_val   = pd.DataFrame(scaler.transform(X_val),       columns=FEATURE_COLS)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=FEATURE_COLS)

    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def train_model(X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
    """
    Train a LightGBM classifier with early stopping on the validation set.
    LightGBM is preferred over sklearn GBM for:
        - Speed on large datasets
        - Built-in early stopping
        - Native feature importance
    """
    model = lgb.LGBMClassifier(
        n_estimators      = 500,
        learning_rate     = 0.05,
        max_depth         = 6,
        num_leaves        = 31,
        min_child_samples = 30,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        random_state      = 42,
        verbose           = -1,
    )

    model.fit(
        X_train, y_train,
        eval_set          = [(X_val, y_val)],
        callbacks         = [lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(period=100)],
    )

    logger.info(f"Best iteration: {model.best_iteration_}")
    return model


def evaluate_model(model, X_test, y_test, save: bool = True) -> dict:
    """
    Full evaluation suite:
        - ROC-AUC (primary metric)
        - Classification report (precision, recall, F1)
        - Confusion matrix
        - ROC curve plot
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    logger.info(f"Test ROC-AUC: {auc:.4f}")
    print(f"\n{'='*50}")
    print(f"  Test ROC-AUC : {auc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Low Engagement", "High Engagement"]))

    # ── ROC Curve ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0], color="#2171B5")
    axes[0].set_title(f"ROC Curve (AUC = {auc:.3f})", fontweight="bold")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)

    # ── Confusion Matrix ─────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Low Eng.", "High Eng."]).plot(
        ax=axes[1], colorbar=False, cmap="Blues"
    )
    axes[1].set_title("Confusion Matrix", fontweight="bold")

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "model_evaluation.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved evaluation plot → {path}")
    plt.show()

    return {"roc_auc": auc, "y_prob": y_prob, "y_pred": y_pred}


def plot_feature_importance(model, feature_names: list, save: bool = True) -> None:
    """
    Plot LightGBM feature importance (gain-based).
    Gain importance reflects how much each feature contributes to
    reducing prediction error — more interpretable than split count.
    """
    importance = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(importance["feature"], importance["importance"],
            color="#2171B5", edgecolor="white")
    ax.set_title("Feature Importance (Gain)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved feature importance plot → {path}")
    plt.show()


def cross_validate(features: pd.DataFrame, k: int = 5) -> float:
    """
    Stratified K-fold cross-validation to confirm model stability.
    Returns mean ROC-AUC across folds.
    """
    df = features.dropna(subset=FEATURE_COLS + [TARGET])
    X = StandardScaler().fit_transform(df[FEATURE_COLS])
    y = df[TARGET]

    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                random_state=42, verbose=-1)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    logger.info(f"CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"Cross-validation ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()


def run_retention_pipeline(features: pd.DataFrame) -> dict:
    """Full train → evaluate pipeline. Returns evaluation metrics."""
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(features)
    model   = train_model(X_train, y_train, X_val, y_val)
    metrics = evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, FEATURE_COLS)
    cross_validate(features)
    return {"model": model, "scaler": scaler, **metrics}


if __name__ == "__main__":
    from data_loader import load_interactions
    from feature_engineering import build_user_features
    interactions = load_interactions(sample_users=50_000)
    features     = build_user_features(interactions)
    results      = run_retention_pipeline(features)
    print(f"Final ROC-AUC: {results['roc_auc']:.4f}")
