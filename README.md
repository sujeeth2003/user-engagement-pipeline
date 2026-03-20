# User Engagement Analytics & Retention Modeling Pipeline

An end-to-end data engineering and ML pipeline that turns raw user interaction logs into actionable behavioral segments and a retention prediction model — built on the MyAnimeList dataset (70k+ users × 12k+ items).

The core challenge this project solves: **raw interaction logs tell you what happened, but not why users stay or leave.** This pipeline transforms those logs into structured behavioral features, surfaces distinct user segments, and trains a model that predicts high-engagement users with **ROC-AUC ≈ 0.87**.

---

## What This Pipeline Does

```
Raw interaction logs (70k users × 12k items)
        │
        ▼
┌─────────────────────────────────┐
│  1. Data Loading & Validation   │  Schema checks, user sampling, type coercion
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  2. Feature Engineering         │  Completion rate, drop rate, scoring rate,
│                                 │  rewatch rate, interaction intensity, score std
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  3. Cohort Analysis             │  New / Growing / Retained lifecycle cohorts;
│                                 │  engagement trend plots per cohort
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  4. K-Means User Segmentation   │  Elbow + silhouette K selection;
│                                 │  PCA projection; radar chart profiles
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  5. Retention Model (LightGBM)  │  Stratified train/val/test split;
│                                 │  early stopping; ROC-AUC ≈ 0.87;
│                                 │  5-fold cross-validation
└─────────────────────────────────┘
        │
        ▼
  outputs/ (plots, logs, metrics)
```

---

## Key Results

| Metric | Value |
|---|---|
| Users processed | 70,000+ |
| Items in dataset | 12,000+ |
| Retention model ROC-AUC | **≈ 0.87** |
| User segments identified | 4 distinct behavioral personas |
| Top predictive feature | Completion rate |

---

## Features Engineered

| Feature | What It Measures |
|---|---|
| `completion_rate` | Fraction of started titles the user finished |
| `drop_rate` | Fraction of titles abandoned mid-way |
| `scoring_rate` | How often users leave an explicit rating |
| `rewatch_rate` | Deep loyalty signal — rewatching finished content |
| `log_episodes_watched` | Total consumption volume (log-scaled) |
| `score_std` | Consistency of taste vs eclectic preferences |
| `mean_score` | Generosity vs critical rating tendency |

These features mirror what a streaming platform would compute to understand **member engagement depth and churn risk**.

---

## User Segments (K-Means, K=4)

| Segment | Profile |
|---|---|
| **Power Completionists** | High completion rate, high volume, low drop rate |
| **Engaged Critics** | High scoring rate, frequent rewatchers |
| **Selective Browsers** | High drop rate, low completion, exploratory |
| **Casual Explorers** | Low volume, moderate engagement across metrics |

---

## Project Structure

```
user-engagement-pipeline/
├── src/
│   ├── data_loader.py          # Load & validate MyAnimeList dataset
│   ├── feature_engineering.py  # Per-user behavioral features + cohort labels
│   ├── cohort_analysis.py      # Cohort summary + engagement distribution plots
│   ├── clustering.py           # K-Means segmentation + PCA + radar charts
│   └── retention_model.py      # LightGBM classifier + evaluation suite
├── notebooks/
│   └── full_pipeline_walkthrough.ipynb   # Narrative walkthrough of every step
├── data/
│   ├── raw/                    # Place downloaded dataset files here (not tracked)
│   └── processed/              # Intermediate feature tables
├── outputs/                    # All plots and logs saved here
├── run_pipeline.py             # Single entry-point to run everything
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download from Kaggle: [MyAnimeList Dataset](https://www.kaggle.com/datasets/azathoth42/myanimelist)

Place these files in `data/raw/`:
- `UserAnimeList.csv`
- `AnimeList.csv`
- `UserList.csv`

### 3. Run the full pipeline
```bash
python run_pipeline.py --users 70000 --clusters 4
```

To let the pipeline find the optimal K automatically:
```bash
python run_pipeline.py --find-k
```

### 4. Or walk through the notebook
```bash
jupyter notebook notebooks/full_pipeline_walkthrough.ipynb
```

---

## Design Decisions

**Why LightGBM over sklearn GBM?**
Speed on large datasets, built-in early stopping, and more interpretable gain-based feature importance. On 50k+ users, LightGBM trains 10–20× faster.

**Why log-scale episodes watched?**
Raw episode counts are heavily right-skewed (power users watch 10,000+ episodes). Log transform brings the distribution closer to normal and prevents the model from over-indexing on extreme outliers.

**Why stratified splits?**
The target class (high_engagement) is imbalanced at ~35%. Stratified splitting ensures train/val/test sets have the same class ratio, giving honest AUC estimates.

**Why cohort analysis before modeling?**
Understanding how engagement varies by user lifecycle stage reveals *where* to intervene — not just *who* to target. Retained users and new users need completely different nudges.

---

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, LightGBM
- **Visualization** — matplotlib, seaborn
- **ML** — LightGBM (gradient boosting), K-Means, PCA
- **Data Engineering** — modular ETL pipeline, schema validation, stratified sampling

---

## Author

[Sujeeth Sukumar](https://www.linkedin.com/in/sujeeth73) · [Portfolio](https://sujeeth2003.github.io/Portfolio/)
