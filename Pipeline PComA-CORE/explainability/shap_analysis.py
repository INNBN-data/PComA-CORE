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
import shap

from preprocessing.feature_encoding import eligible_rows, select_features


def main():
    df = pd.read_csv("data/features_development.csv", low_memory=False)
    mask = eligible_rows(df, "pcoma_c", "derived_high_complexity_case")
    eligible = df.loc[mask].copy()
    X = select_features(eligible, "pcoma_c")
    bundle = joblib.load("models/pcoma_c/pcoma_c_model.joblib")
    X_transformed = bundle["preprocessor"].transform(X)
    names = bundle.get("transformed_features") or [f"feature_{i}" for i in range(X_transformed.shape[1])]

    max_rows = min(500, len(X_transformed))
    sample = X_transformed[:max_rows]
    explainer = shap.TreeExplainer(bundle["model"])
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[-1]

    destination = Path("explainability/generated")
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "pcoma_c_shap_values.npy", np.asarray(values))
    shap.summary_plot(values, sample, feature_names=names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(destination / "pcoma_c_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("PComA-C SHAP analysis complete.")


if __name__ == "__main__":
    main()
