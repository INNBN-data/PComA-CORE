from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sksurv.metrics import concordance_index_censored

from models.common import save_json
from preprocessing.feature_encoding import eligible_rows, select_features


def main():
    df = pd.read_csv("data/features_temporal.csv", low_memory=False)
    mask = eligible_rows(df, "pcoma_o_aft")
    eligible = df.loc[mask].dropna(subset=["pcoma_o_event", "pcoma_o_time_days"]).copy()
    X = select_features(eligible, "pcoma_o_aft")
    event = pd.to_numeric(eligible["pcoma_o_event"], errors="raise").astype(int)
    time = pd.to_numeric(eligible["pcoma_o_time_days"], errors="raise").astype(float)

    output = Path("models/pcoma_o/aft")
    preprocessor = joblib.load(output / "preprocessor.joblib")
    booster = xgb.Booster()
    booster.load_model(output / "pcoma_o_aft_model.json")
    X_transformed = preprocessor.transform(X)
    predicted_time = booster.predict(xgb.DMatrix(X_transformed))
    risk = -np.asarray(predicted_time, dtype=float)
    c_index = concordance_index_censored(
        event.astype(bool).to_numpy(), time.to_numpy(), risk
    )[0]

    id_columns = [column for column in [
        "study_patient_id", "study_case_id", "aneurysm_id", "procedure_id", "index_year"
    ] if column in eligible.columns]
    predictions = eligible[id_columns].copy()
    predictions["event"] = event.to_numpy()
    predictions["time_days"] = time.to_numpy()
    predictions["risk_score"] = risk
    predictions.to_csv(output / "temporal_predictions.csv", index=False)

    metrics = {
        "n": int(len(eligible)),
        "events": int(event.sum()),
        "c_index": float(c_index),
        "interpretation": "temporal stress test; sparse contemporary non-recovery limits standalone inference",
    }
    save_json(output / "temporal_metrics.json", metrics)
    print(f"PComA-O AFT temporal stress-test metrics: {metrics}")


if __name__ == "__main__":
    main()
