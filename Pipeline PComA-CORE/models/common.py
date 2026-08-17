from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def patient_groups(df: pd.DataFrame) -> pd.Series:
    group_column = load_yaml("config/model_settings.yaml")["general"]["group_column"]
    if group_column not in df.columns:
        raise KeyError(f"{group_column} is required for patient-grouped resampling.")
    groups = df[group_column].astype("string")
    if groups.isna().any() or groups.str.strip().eq("").any():
        raise ValueError(f"{group_column} contains missing or empty values.")
    return groups.astype(str)


def choose_youden_threshold(y_true, probabilities) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.concatenate(([0.0], p, [1.0])))
    best_threshold = 0.5
    best_value = -np.inf
    for threshold in candidates:
        predicted = (p >= threshold).astype(int)
        tp = ((predicted == 1) & (y == 1)).sum()
        tn = ((predicted == 0) & (y == 0)).sum()
        fp = ((predicted == 1) & (y == 0)).sum()
        fn = ((predicted == 0) & (y == 1)).sum()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        value = sensitivity + specificity - 1.0
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold


def binary_metrics(y_true, probabilities, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predicted = (p >= threshold).astype(int)
    tp = int(((predicted == 1) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def ordinal_concordance(y_true, predicted_score) -> float:
    y = np.asarray(y_true, dtype=float)
    score = np.asarray(predicted_score, dtype=float)
    concordant = 0.0
    comparable = 0
    for i in range(len(y) - 1):
        delta_y = y[i + 1:] - y[i]
        mask = delta_y != 0
        if not mask.any():
            continue
        delta_s = score[i + 1:][mask] - score[i]
        direction = np.sign(delta_y[mask]) * np.sign(delta_s)
        concordant += float((direction > 0).sum()) + 0.5 * float((direction == 0).sum())
        comparable += int(mask.sum())
    return concordant / comparable if comparable else float("nan")


def ranked_probability_score(y_true, probabilities) -> float:
    y = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    cumulative_pred = np.cumsum(probs[:, :-1], axis=1)
    thresholds = np.arange(probs.shape[1] - 1)
    cumulative_obs = (y[:, None] <= thresholds[None, :]).astype(float)
    return float(np.mean(np.sum((cumulative_pred - cumulative_obs) ** 2, axis=1) / (probs.shape[1] - 1)))


def ordinal_metrics(y_true, probabilities) -> dict:
    y = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 7:
        raise ValueError("PComA-E ordinal probabilities must have seven columns for mRS 0-6.")
    predicted_class = np.argmax(probs, axis=1)
    expected_score = probs @ np.arange(7, dtype=float)
    concordance = ordinal_concordance(y, expected_score)
    return {
        "n": int(len(y)),
        "ordinal_concordance": float(concordance),
        "somers_dxy": float(2.0 * concordance - 1.0),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y, predicted_class, weights="quadratic")
        ),
        "ranked_probability_score": ranked_probability_score(y, probs),
        "average_absolute_ordinal_error": float(np.mean(np.abs(y - predicted_class))),
    }


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)


def save_predictions(path: str | Path, df: pd.DataFrame, observed, probability, **extra) -> None:
    columns = [column for column in [
        "study_patient_id", "study_case_id", "aneurysm_id", "procedure_id", "index_year"
    ] if column in df.columns]
    out = df.loc[:, columns].copy()
    out["observed"] = np.asarray(observed)
    out["probability"] = np.asarray(probability, dtype=float)
    for key, value in extra.items():
        out[key] = value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
