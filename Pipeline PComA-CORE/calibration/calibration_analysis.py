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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from models.common import save_json

MODELS = {
    "pcoma_c": Path("models/pcoma_c"),
    "pcoma_r": Path("models/pcoma_r"),
    "pcoma_e_binary": Path("models/pcoma_e"),
}


def logit(probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def main():
    summary = {}
    generated = Path("calibration/generated")
    generated.mkdir(parents=True, exist_ok=True)

    for model_key, model_dir in MODELS.items():
        labels_path = model_dir / "oof_labels.npy"
        probabilities_path = model_dir / "oof_proba.npy"
        temporal_path = model_dir / "temporal_predictions.csv"
        if not labels_path.exists() or not probabilities_path.exists():
            print(f"Skipping {model_key}: OOF files are absent.")
            continue
        y = np.load(labels_path).astype(int)
        p = np.load(probabilities_path).astype(float)
        recalibrator = LogisticRegression(C=1e6, solver="lbfgs")
        recalibrator.fit(logit(p), y)
        p_cal = recalibrator.predict_proba(logit(p))[:, 1]
        metrics = {
            "development_raw_brier": float(brier_score_loss(y, p)),
            "development_recalibrated_brier": float(brier_score_loss(y, p_cal)),
            "recalibration_intercept": float(recalibrator.intercept_[0]),
            "recalibration_slope": float(recalibrator.coef_[0, 0]),
            "calibrator_source": "development out-of-fold predictions",
        }
        joblib.dump(recalibrator, generated / f"{model_key}_logistic_recalibrator.joblib")

        if temporal_path.exists():
            temporal = pd.read_csv(temporal_path)
            raw = temporal["probability"].to_numpy(dtype=float)
            observed = temporal["observed"].to_numpy(dtype=int)
            calibrated = recalibrator.predict_proba(logit(raw))[:, 1]
            temporal["calibrated_probability"] = calibrated
            temporal.to_csv(generated / f"{model_key}_temporal_calibrated.csv", index=False)
            metrics.update({
                "temporal_raw_brier": float(brier_score_loss(observed, raw)),
                "temporal_recalibrated_brier": float(brier_score_loss(observed, calibrated)),
            })
        summary[model_key] = metrics
        print(model_key, metrics)

    save_json(generated / "calibration_summary.json", summary)


if __name__ == "__main__":
    main()
