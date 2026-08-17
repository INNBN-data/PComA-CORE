from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from models.common import (
    binary_metrics,
    choose_youden_threshold,
    load_yaml,
    patient_groups,
    save_json,
    save_predictions,
)
from preprocessing.feature_encoding import (
    build_preprocessor,
    eligible_rows,
    select_features,
    transformed_feature_names,
)


def train_grouped_binary(
    df: pd.DataFrame,
    model_key: str,
    target: str,
    estimator,
    output_dir: str,
    scale_numeric: bool = False,
) -> dict:
    mask = eligible_rows(df, model_key, target)
    eligible = df.loc[mask].copy()
    X = select_features(eligible, model_key)
    y = pd.to_numeric(eligible[target], errors="raise").astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"{target} must contain only 0/1 among eligible rows.")
    groups = patient_groups(eligible)

    seed = int(load_yaml("config/random_seeds.yaml")["global_seed"])
    n_splits = int(load_yaml("config/random_seeds.yaml")["outer_cv_splits"])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(eligible), np.nan, dtype=float)
    fold_ids = np.full(len(eligible), -1, dtype=int)

    for fold, (train_idx, validation_idx) in enumerate(splitter.split(X, y, groups), start=1):
        preprocessor = build_preprocessor(X.iloc[train_idx], scale_numeric=scale_numeric)
        X_train = preprocessor.fit_transform(X.iloc[train_idx])
        X_validation = preprocessor.transform(X.iloc[validation_idx])
        fold_model = clone(estimator)
        fold_model.fit(X_train, y.iloc[train_idx])
        oof[validation_idx] = fold_model.predict_proba(X_validation)[:, 1]
        fold_ids[validation_idx] = fold
        print(f"{model_key}: completed grouped fold {fold}.")

    if np.isnan(oof).any() or (fold_ids < 0).any():
        raise RuntimeError(f"{model_key} OOF predictions are incomplete.")

    threshold = choose_youden_threshold(y, oof)
    metrics = binary_metrics(y, oof, threshold=threshold)
    metrics["grouped_by"] = "study_patient_id"
    metrics["validation"] = "five-fold out-of-fold reference implementation"

    final_preprocessor = build_preprocessor(X, scale_numeric=scale_numeric)
    X_full = final_preprocessor.fit_transform(X)
    final_model = clone(estimator)
    final_model.fit(X_full, y)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_key": model_key,
        "target": target,
        "preprocessor": final_preprocessor,
        "model": final_model,
        "threshold": threshold,
        "raw_features": list(X.columns),
        "transformed_features": transformed_feature_names(final_preprocessor),
    }
    joblib.dump(bundle, output / f"{model_key}_model.joblib")
    np.save(output / "oof_proba.npy", oof)
    np.save(output / "oof_labels.npy", y.to_numpy())
    np.save(output / "oof_fold.npy", fold_ids)
    save_predictions(
        output / "oof_predictions.csv",
        eligible,
        observed=y,
        probability=oof,
        fold=fold_ids,
        model_key=model_key,
    )
    save_json(output / "metrics.json", metrics)
    return metrics


def evaluate_grouped_binary(
    df: pd.DataFrame,
    model_key: str,
    target: str,
    model_path: str,
    output_dir: str,
) -> dict:
    mask = eligible_rows(df, model_key, target)
    eligible = df.loc[mask].copy()
    X = select_features(eligible, model_key)
    y = pd.to_numeric(eligible[target], errors="raise").astype(int)
    bundle = joblib.load(model_path)
    X_transformed = bundle["preprocessor"].transform(X)
    probabilities = bundle["model"].predict_proba(X_transformed)[:, 1]
    threshold = float(bundle["threshold"])
    metrics = binary_metrics(y, probabilities, threshold=threshold)
    metrics["threshold_source"] = "development OOF"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_predictions(
        output / "temporal_predictions.csv",
        eligible,
        observed=y,
        probability=probabilities,
        model_key=model_key,
    )
    save_json(output / "temporal_metrics.json", metrics)
    return metrics
