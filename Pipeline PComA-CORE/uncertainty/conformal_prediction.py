from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd

from models.common import load_yaml, save_json

MODELS = {
    "pcoma_c": Path("models/pcoma_c"),
    "pcoma_r": Path("models/pcoma_r"),
    "pcoma_e_binary": Path("models/pcoma_e"),
}


def conformal_quantile(scores, alpha):
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def main():
    alpha = float(load_yaml("config/random_seeds.yaml")["conformal_alpha"])
    destination = Path("uncertainty/generated")
    destination.mkdir(parents=True, exist_ok=True)
    summary = {}

    for model_key, model_dir in MODELS.items():
        labels_path = model_dir / "oof_labels.npy"
        probability_path = model_dir / "oof_proba.npy"
        temporal_path = model_dir / "temporal_predictions.csv"
        if not (labels_path.exists() and probability_path.exists() and temporal_path.exists()):
            print(f"Skipping {model_key}: required prediction files are absent.")
            continue
        y_oof = np.load(labels_path).astype(int)
        p_oof = np.load(probability_path).astype(float)
        true_probability = np.where(y_oof == 1, p_oof, 1.0 - p_oof)
        qhat = conformal_quantile(1.0 - true_probability, alpha)

        temporal = pd.read_csv(temporal_path)
        observed = temporal["observed"].to_numpy(dtype=int)
        p1 = temporal["probability"].to_numpy(dtype=float)
        p0 = 1.0 - p1
        include0 = (1.0 - p0) <= qhat
        include1 = (1.0 - p1) <= qhat
        empty = ~(include0 | include1)
        include0[empty & (p0 >= p1)] = True
        include1[empty & (p1 > p0)] = True
        covered = np.where(observed == 0, include0, include1)
        set_size = include0.astype(int) + include1.astype(int)

        out = temporal.copy()
        out["include_class_0"] = include0.astype(int)
        out["include_class_1"] = include1.astype(int)
        out["prediction_set_size"] = set_size
        out.to_csv(destination / f"{model_key}_temporal_prediction_sets.csv", index=False)
        metrics = {
            "alpha": alpha,
            "nominal_coverage": 1.0 - alpha,
            "qhat": qhat,
            "temporal_n": int(len(out)),
            "temporal_coverage": float(np.mean(covered)),
            "average_prediction_set_size": float(np.mean(set_size)),
            "method": "cross-fitted development nonconformity scores applied unchanged to temporal predictions",
        }
        summary[model_key] = metrics
        print(model_key, metrics)

    save_json(destination / "conformal_summary.json", summary)


if __name__ == "__main__":
    main()
