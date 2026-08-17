from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from models.common import load_yaml, patient_groups, save_json
from preprocessing.feature_encoding import build_preprocessor, eligible_rows, select_features


def main():
    df = pd.read_csv("data/features_development.csv", low_memory=False)
    mask = eligible_rows(df, "pcoma_c", "derived_high_complexity_case")
    eligible = df.loc[mask].copy()
    X = select_features(eligible, "pcoma_c")
    y = pd.to_numeric(eligible["derived_high_complexity_case"], errors="raise").astype(int)
    groups = patient_groups(eligible)

    seeds = load_yaml("config/random_seeds.yaml")
    outer = StratifiedGroupKFold(
        n_splits=int(seeds["outer_cv_splits"]), shuffle=True,
        random_state=int(seeds["global_seed"]),
    )
    scores = []
    selected = []

    for fold, (train_idx, validation_idx) in enumerate(outer.split(X, y, groups), start=1):
        inner = StratifiedGroupKFold(
            n_splits=int(seeds["inner_cv_splits"]), shuffle=True,
            random_state=int(seeds["global_seed"]) + fold,
        )
        preprocessor = build_preprocessor(X.iloc[train_idx], scale_numeric=False)
        base = load_yaml("config/hyperparameters.yaml")["catboost"].copy()
        base["iterations"] = 1000
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", CatBoostClassifier(**base)),
        ])
        grid = {
            "model__depth": [4, 6, 8],
            "model__learning_rate": [0.03, 0.05, 0.10],
        }
        search = GridSearchCV(
            pipeline, grid, scoring="roc_auc", cv=inner, n_jobs=1, refit=True
        )
        search.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            groups=groups.iloc[train_idx],
        )
        probabilities = search.best_estimator_.predict_proba(X.iloc[validation_idx])[:, 1]
        score = roc_auc_score(y.iloc[validation_idx], probabilities)
        scores.append(float(score))
        selected.append(search.best_params_)
        print(f"Outer fold {fold}: AUC={score:.3f}; {search.best_params_}")

    output = {
        "mean_auc": float(np.mean(scores)),
        "sd_auc": float(np.std(scores, ddof=1)),
        "fold_auc": scores,
        "selected_parameters": selected,
        "grouped_by": "study_patient_id",
        "note": "Reference grid search; exact manuscript tuning requires the locked Bayesian-optimisation objects.",
    }
    save_json("validation/generated/nested_cv_pcoma_c.json", output)
    print(output)


if __name__ == "__main__":
    main()
