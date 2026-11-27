"""FrameLens Member 2: Nov 25–26 baseline experiments.

Run from the repository root: python -m src.train
The held-out test.csv is deliberately NOT read by this module.
"""
import hashlib
import json
import platform
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.evaluate import classification_metrics

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_SPLITS = 5
TFIDF_CONFIG = {"ngram_range": (1, 2), "min_df": 1, "max_df": 1.0,
                "max_features": None, "sublinear_tf": False,
                "stop_words": None, "lowercase": True}
LR_CONFIG = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs",
             "random_state": SEED, "class_weight": None}


def preprocess_text(text: str) -> str:
    """Same normalization as Member 1's Clean.ipynb; for raw demo input."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_training_data():
    """Use the existing locked training split; never resplit the full dataset."""
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    required = ["model_text", "label", "news_link", "original_text"]
    if not set(required).issubset(train_df.columns):
        raise ValueError("Training CSV is missing required columns.")
    if train_df[required].isna().any().any():
        raise ValueError("Training data has missing required values.")
    if set(train_df["label"]) != {0, 1}:
        raise ValueError("Expected labels: 0 = Non-biased, 1 = Biased.")
    if not train_df["model_text"].str.strip().ne("").all():
        raise ValueError("Empty model text found.")
    if not train_df["original_text"].map(preprocess_text).equals(train_df["model_text"]):
        raise ValueError("Preprocessing does not match Member 1's exported model_text.")
    return train_df


def build_baselines():
    """Create fresh, unfitted estimators; TF-IDF belongs inside the pipeline."""
    majority = Pipeline([
        ("classifier", DummyClassifier(strategy="most_frequent"))
    ])
    logistic = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
        ("classifier", LogisticRegression(**LR_CONFIG))
    ])
    return {"Majority baseline": majority, "TF-IDF + Logistic Regression": logistic}


def evaluate_baselines(models, X, y, groups):
    """Evaluate both models using identical grouped CV folds from train.csv."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(cv.split(X, y, groups=groups))
    rows = []
    for fold, (fit_idx, val_idx) in enumerate(splits, start=1):
        # Article URLs are used only to group the split, never as input features.
        if set(groups.iloc[fit_idx]) & set(groups.iloc[val_idx]):
            raise ValueError("An article appears in both fitting and validation data.")
        for name, estimator in models.items():
            model = clone(estimator)
            model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            predictions = model.predict(X.iloc[val_idx])
            scores = classification_metrics(y.iloc[val_idx], predictions)
            rows.append({"model": name, "fold": fold,
                         "fit_rows": len(fit_idx), "validation_rows": len(val_idx),
                         "validation_non_biased": int((y.iloc[val_idx] == 0).sum()),
                         "validation_biased": int((y.iloc[val_idx] == 1).sum()),
                         **scores})
    fold_results = pd.DataFrame(rows)
    metrics = list(classification_metrics([0, 1], [0, 1]))
    summary = fold_results.groupby("model", sort=False)[metrics].mean()
    summary["macro_f1_std"] = fold_results.groupby("model", sort=False)["macro_f1"].std(ddof=0)
    return fold_results, summary.reset_index()


def save_artifacts(models, X, y, fold_results, summary):
    """Save preliminary metrics and fit a demo pipeline on all training rows."""
    reports = ROOT / "reports"
    model_dir = ROOT / "models"
    reports.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    fold_results.to_csv(reports / "baseline_fold_metrics.csv", index=False, float_format="%.6f")
    summary.to_csv(reports / "baseline_results.csv", index=False, float_format="%.6f")

    logistic = clone(models["TF-IDF + Logistic Regression"])
    logistic.fit(X, y)
    model_path = model_dir / "baseline_logistic_regression.joblib"
    joblib.dump(logistic, model_path, compress=3)

    config = {
        "stage": "November 25–26: preliminary baselines, not final test evaluation",
        "input": "data/train.csv: model_text", "label": "0=Non-biased; 1=Biased",
        "train_sha256": hashlib.sha256((ROOT / "data/train.csv").read_bytes()).hexdigest(),
        "training_rows": len(X), "training_label_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "cv": {"method": "StratifiedGroupKFold", "n_splits": N_SPLITS,
               "shuffle": True, "random_state": SEED, "groups": "news_link"},
        "majority": {"strategy": "most_frequent"},
        "tfidf": TFIDF_CONFIG, "logistic_regression": LR_CONFIG,
        "majority_parameters": models["Majority baseline"].named_steps["classifier"].get_params(),
        "tfidf_parameters": logistic.named_steps["tfidf"].get_params(),
        "logistic_parameters": logistic.named_steps["classifier"].get_params(),
        "vocabulary_size_full_train": len(logistic.named_steps["tfidf"].vocabulary_),
        "metric_aggregation": "unweighted mean over five validation folds; population SD for macro F1",
        "held_out_test_evaluated": False,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    (reports / "baseline_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    return model_path


def main():
    train_df = load_training_data()
    X = train_df["model_text"]
    y = train_df["label"]
    groups = train_df["news_link"]
    models = build_baselines()
    folds, summary = evaluate_baselines(models, X, y, groups)
    path = save_artifacts(models, X, y, folds, summary)
    print("Training rows:", len(X))
    print("Training label counts:\n", y.value_counts().to_string())
    print("\nPreliminary validation results (5-fold grouped CV; not test results):")
    print(summary[["model", "accuracy", "macro_f1", "macro_f1_std"]].round(4).to_string(index=False))
    print("\nSaved:", path.relative_to(ROOT))
    print("Held-out test set not evaluated.")


if __name__ == "__main__":
    main()
