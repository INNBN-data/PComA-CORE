from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold

from models.common import load_yaml, patient_groups
from preprocessing.feature_encoding import (
    build_preprocessor, eligible_rows, select_features, transformed_feature_names
)


def main():
    df = pd.read_csv("data/features_development.csv", low_memory=False)
    mask = eligible_rows(df, "pcoma_c", "derived_high_complexity_case")
    eligible = df.loc[mask].copy()
    X = select_features(eligible, "pcoma_c")
    y = pd.to_numeric(eligible["derived_high_complexity_case"], errors="raise").astype(int)
    groups = patient_groups(eligible)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2024)
    by_feature = defaultdict(list)

    for fold, (train_idx, validation_idx) in enumerate(splitter.split(X, y, groups), start=1):
        preprocessor = build_preprocessor(X.iloc[train_idx], scale_numeric=False)
        Xt = preprocessor.fit_transform(X.iloc[train_idx])
        model = CatBoostClassifier(**load_yaml("config/hyperparameters.yaml")["catboost"])
        model.fit(Xt, y.iloc[train_idx])
        names = transformed_feature_names(preprocessor)
        for name, value in zip(names, model.feature_importances_):
            by_feature[name].append(float(value))
        print(f"Completed feature-stability fold {fold}.")

    records = []
    for feature, values in by_feature.items():
        padded = values + [0.0] * (5 - len(values))
        mean = float(np.mean(padded))
        sd = float(np.std(padded, ddof=1))
        records.append({
            "feature": feature,
            "mean_importance": mean,
            "sd_importance": sd,
            "stability_ratio": mean / (sd + 1e-12),
            "folds_present": len(values),
        })
    result = pd.DataFrame(records).sort_values("mean_importance", ascending=False)
    destination = Path("explainability/generated")
    destination.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination / "pcoma_c_feature_stability.csv", index=False)
    print(result.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
