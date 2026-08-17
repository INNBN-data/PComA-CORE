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

from models.common import ordinal_metrics, save_json
from preprocessing.feature_encoding import eligible_rows, select_features
from models.pcoma_e.train_pcoma_e_ordinal import expand_probabilities


def main():
    df = pd.read_csv("data/features_temporal.csv", low_memory=False)
    mask = eligible_rows(df, "pcoma_e_ordinal", "mrs_90d")
    eligible = df.loc[mask].copy()
    X = select_features(eligible, "pcoma_e_ordinal")
    y = pd.to_numeric(eligible["mrs_90d"], errors="raise").astype(int)

    output = Path("models/pcoma_e/ordinal")
    bundle = joblib.load(output / "pcoma_e_ordinal_model.joblib")
    X_transformed = bundle["preprocessor"].transform(X)
    X_transformed = bundle["selector"].transform(X_transformed)
    predicted = bundle["result"].model.predict(
        bundle["result"].params, exog=X_transformed
    )
    probabilities = expand_probabilities(predicted, bundle["classes"])
    metrics = ordinal_metrics(y, probabilities)

    id_columns = [column for column in [
        "study_patient_id", "study_case_id", "aneurysm_id", "procedure_id", "index_year"
    ] if column in eligible.columns]
    predictions = eligible[id_columns].copy()
    predictions["observed_mrs"] = y.to_numpy()
    for category in range(7):
        predictions[f"prob_mrs_{category}"] = probabilities[:, category]
    predictions.to_csv(output / "temporal_predictions.csv", index=False)
    save_json(output / "temporal_metrics.json", metrics)
    print(f"PComA-E ordinal temporal metrics: {metrics}")


if __name__ == "__main__":
    main()
