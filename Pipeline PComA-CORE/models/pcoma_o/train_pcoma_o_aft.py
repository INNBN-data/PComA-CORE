from __future__ import annotations

from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sksurv.metrics import concordance_index_censored

from models.common import load_yaml, patient_groups, save_json
from preprocessing.feature_encoding import (
    build_preprocessor,
    eligible_rows,
    select_features,
    transformed_feature_names,
)


def make_dmatrix(X, time, event):
    matrix = xgb.DMatrix(X)
    lower = np.asarray(time, dtype=float)
    upper = np.where(np.asarray(event, dtype=int) == 1, lower, np.inf)
    matrix.set_float_info("label_lower_bound", lower)
    matrix.set_float_info("label_upper_bound", upper)
    return matrix


def train_booster(X_train, time_train, event_train, X_validation=None, time_validation=None, event_validation=None):
    raw = load_yaml("config/hyperparameters.yaml")["xgboost_aft"].copy()
    num_boost_round = int(raw.pop("num_boost_round"))
    early_stopping_rounds = int(raw.pop("early_stopping_rounds"))
    dtrain = make_dmatrix(X_train, time_train, event_train)
    evals = [(dtrain, "train")]
    kwargs = {}
    if X_validation is not None:
        dvalidation = make_dmatrix(X_validation, time_validation, event_validation)
        evals.append((dvalidation, "validation"))
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    booster = xgb.train(
        params=raw,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        verbose_eval=False,
        **kwargs,
    )
    return booster


def main():
    df = pd.read_csv("data/features_development.csv", low_memory=False)
    mask = eligible_rows(df, "pcoma_o_aft")
    eligible = df.loc[mask].dropna(subset=["pcoma_o_event", "pcoma_o_time_days"]).copy()
    if eligible.empty:
        raise ValueError("No eligible PComA-O observations with valid event and censoring time.")

    X = select_features(eligible, "pcoma_o_aft")
    event = pd.to_numeric(eligible["pcoma_o_event"], errors="raise").astype(int)
    time = pd.to_numeric(eligible["pcoma_o_time_days"], errors="raise").astype(float)
    if not set(event.unique()).issubset({0, 1}):
        raise ValueError("pcoma_o_event must contain only 0/1.")
    if (time <= 0).any():
        raise ValueError("pcoma_o_time_days must be greater than zero.")
    groups = patient_groups(eligible)

    seeds = load_yaml("config/random_seeds.yaml")
    splitter = StratifiedGroupKFold(
        n_splits=int(seeds["outer_cv_splits"]),
        shuffle=True,
        random_state=int(seeds["global_seed"]),
    )
    oof_risk = np.full(len(eligible), np.nan, dtype=float)
    fold_ids = np.full(len(eligible), -1, dtype=int)
    fold_c_indices = []
    best_rounds = []

    for fold, (train_idx, validation_idx) in enumerate(
        splitter.split(X, event, groups), start=1
    ):
        preprocessor = build_preprocessor(X.iloc[train_idx], scale_numeric=False)
        X_train = preprocessor.fit_transform(X.iloc[train_idx])
        X_validation = preprocessor.transform(X.iloc[validation_idx])
        booster = train_booster(
            X_train,
            time.iloc[train_idx],
            event.iloc[train_idx],
            X_validation,
            time.iloc[validation_idx],
            event.iloc[validation_idx],
        )
        predicted_time = booster.predict(xgb.DMatrix(X_validation))
        risk = -np.asarray(predicted_time, dtype=float)
        oof_risk[validation_idx] = risk
        fold_ids[validation_idx] = fold
        c_index = concordance_index_censored(
            event.iloc[validation_idx].astype(bool).to_numpy(),
            time.iloc[validation_idx].to_numpy(),
            risk,
        )[0]
        fold_c_indices.append(float(c_index))
        best_rounds.append(int(getattr(booster, "best_iteration", 999)) + 1)
        print(f"PComA-O AFT fold {fold}: C-index={c_index:.3f}")

    if np.isnan(oof_risk).any():
        raise RuntimeError("PComA-O AFT OOF predictions are incomplete.")
    overall = concordance_index_censored(
        event.astype(bool).to_numpy(), time.to_numpy(), oof_risk
    )[0]

    final_preprocessor = build_preprocessor(X, scale_numeric=False)
    X_full = final_preprocessor.fit_transform(X)
    raw = load_yaml("config/hyperparameters.yaml")["xgboost_aft"].copy()
    raw.pop("early_stopping_rounds")
    configured_rounds = int(raw.pop("num_boost_round"))
    final_rounds = int(np.median(best_rounds)) if best_rounds else configured_rounds
    final_booster = xgb.train(
        params=raw,
        dtrain=make_dmatrix(X_full, time, event),
        num_boost_round=final_rounds,
        evals=[],
        verbose_eval=False,
    )

    output = Path("models/pcoma_o/aft")
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_preprocessor, output / "preprocessor.joblib")
    final_booster.save_model(output / "pcoma_o_aft_model.json")
    np.save(output / "oof_risk.npy", oof_risk)
    np.save(output / "oof_event.npy", event.to_numpy())
    np.save(output / "oof_time.npy", time.to_numpy())
    np.save(output / "oof_fold.npy", fold_ids)

    id_columns = [column for column in [
        "study_patient_id", "study_case_id", "aneurysm_id", "procedure_id", "index_year"
    ] if column in eligible.columns]
    predictions = eligible[id_columns].copy()
    predictions["event"] = event.to_numpy()
    predictions["time_days"] = time.to_numpy()
    predictions["risk_score"] = oof_risk
    predictions["fold"] = fold_ids
    predictions.to_csv(output / "oof_predictions.csv", index=False)

    metadata = {
        "model_key": "pcoma_o_aft",
        "eligible_n": int(len(eligible)),
        "events": int(event.sum()),
        "oof_c_index": float(overall),
        "fold_c_indices": fold_c_indices,
        "selected_final_rounds": final_rounds,
        "grouped_by": "study_patient_id",
        "transformed_features": transformed_feature_names(final_preprocessor),
        "prediction_orientation": "higher risk score indicates shorter expected recovery time",
    }
    save_json(output / "metrics.json", metadata)
    print(f"PComA-O AFT metrics: {metadata}")


if __name__ == "__main__":
    main()
