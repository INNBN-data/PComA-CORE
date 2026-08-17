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
from sklearn.model_selection import StratifiedGroupKFold

from models.common import load_yaml, patient_groups, save_json
from preprocessing.feature_encoding import build_preprocessor, eligible_rows, select_features


def fit_probabilities(X_train, y_train, X_test):
    preprocessor = build_preprocessor(X_train, scale_numeric=False)
    Xt = preprocessor.fit_transform(X_train)
    Xv = preprocessor.transform(X_test)
    model = CatBoostClassifier(**load_yaml("config/hyperparameters.yaml")["catboost"])
    model.fit(Xt, y_train)
    return model.predict_proba(Xt)[:, 1], model.predict_proba(Xv)[:, 1]


def main():
    df = pd.read_csv("data/features_development.csv", low_memory=False)
    mask = eligible_rows(df, "pcoma_c", "derived_high_complexity_case")
    eligible = df.loc[mask].copy()
    X = select_features(eligible, "pcoma_c")
    y = pd.to_numeric(eligible["derived_high_complexity_case"], errors="raise").astype(int)
    groups = patient_groups(eligible)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2024)
    train_idx, validation_idx = next(splitter.split(X, y, groups))
    training_groups = np.asarray(pd.unique(groups.iloc[train_idx]))
    rng = np.random.default_rng(2024)
    training_groups = rng.permutation(training_groups)
    fractions = load_yaml("config/random_seeds.yaml")["learning_curve_fractions"]

    records = []
    for fraction in fractions:
        n_groups = max(2, int(np.ceil(float(fraction) * len(training_groups))))
        selected = set(training_groups[:n_groups])
        subset_idx = train_idx[groups.iloc[train_idx].isin(selected).to_numpy()]
        if y.iloc[subset_idx].nunique() < 2:
            continue
        train_probability, validation_probability = fit_probabilities(
            X.iloc[subset_idx], y.iloc[subset_idx], X.iloc[validation_idx]
        )
        record = {
            "fraction": float(fraction),
            "patient_groups": int(n_groups),
            "rows": int(len(subset_idx)),
            "train_auc": float(roc_auc_score(y.iloc[subset_idx], train_probability)),
            "validation_auc": float(roc_auc_score(y.iloc[validation_idx], validation_probability)),
        }
        records.append(record)
        print(record)
    save_json("validation/generated/learning_curve_pcoma_c.json", {"records": records})


if __name__ == "__main__":
    main()
