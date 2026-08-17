from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from preprocessing.feature_encoding import eligible_rows, select_features


def predict(bundle, X):
    transformed = bundle["preprocessor"].transform(X)
    return bundle["model"].predict_proba(transformed)[:, 1]


def main():
    df = pd.read_csv("data/features_development.csv", low_memory=False)
    mask = eligible_rows(df, "pcoma_c", "derived_high_complexity_case")
    eligible = df.loc[mask].copy()
    X = select_features(eligible, "pcoma_c")
    y = pd.to_numeric(eligible["derived_high_complexity_case"], errors="raise").astype(int)
    bundle = joblib.load("models/pcoma_c/pcoma_c_model.joblib")
    baseline = roc_auc_score(y, predict(bundle, X))
    rng = np.random.default_rng(2024)
    records = []
    for column in X.columns:
        drops = []
        for _ in range(10):
            permuted = X.copy()
            permuted[column] = rng.permutation(permuted[column].to_numpy())
            drops.append(baseline - roc_auc_score(y, predict(bundle, permuted)))
        records.append({
            "feature": column,
            "mean_auc_drop": float(np.mean(drops)),
            "sd_auc_drop": float(np.std(drops, ddof=1)),
        })
    result = pd.DataFrame(records).sort_values("mean_auc_drop", ascending=False)
    destination = Path("explainability/generated")
    destination.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination / "pcoma_c_permutation_importance.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
