from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.feature_encoding import eligible_rows, select_features


def predict(bundle, X):
    transformed = bundle["preprocessor"].transform(X)
    return bundle["model"].predict_proba(transformed)[:, 1]


def one_dimensional_ale(bundle, X, feature, bins=10):
    values = pd.to_numeric(X[feature], errors="coerce")
    observed = values.dropna()
    if observed.nunique() < 3:
        raise ValueError(f"{feature} has insufficient distinct observed values for ALE.")
    edges = np.unique(np.quantile(observed, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        raise ValueError(f"{feature} produced insufficient quantile bins.")
    bin_index = np.clip(np.digitize(values, edges[1:-1], right=True), 0, len(edges) - 2)
    effects = np.zeros(len(edges) - 1, dtype=float)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for index in range(len(edges) - 1):
        mask = values.notna() & (bin_index == index)
        if not mask.any():
            continue
        lower = X.loc[mask].copy()
        upper = X.loc[mask].copy()
        lower[feature] = edges[index]
        upper[feature] = edges[index + 1]
        effects[index] = np.mean(predict(bundle, upper) - predict(bundle, lower))
        counts[index] = int(mask.sum())
    accumulated = np.cumsum(effects)
    weighted_center = np.average(accumulated, weights=np.maximum(counts, 1))
    accumulated -= weighted_center
    midpoints = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"feature_value": midpoints, "ale": accumulated, "n": counts})


def main():
    feature = "neck_width_mm"
    df = pd.read_csv("data/features_development.csv", low_memory=False)
    mask = eligible_rows(df, "pcoma_c", "derived_high_complexity_case")
    X = select_features(df.loc[mask], "pcoma_c")
    bundle = joblib.load("models/pcoma_c/pcoma_c_model.joblib")
    result = one_dimensional_ale(bundle, X, feature)
    destination = Path("explainability/generated")
    destination.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination / f"pcoma_c_ale_{feature}.csv", index=False)
    plt.figure(figsize=(7, 5))
    plt.plot(result["feature_value"], result["ale"], marker="o")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel(feature)
    plt.ylabel("Accumulated local effect")
    plt.tight_layout()
    plt.savefig(destination / f"pcoma_c_ale_{feature}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
