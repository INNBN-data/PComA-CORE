from __future__ import annotations

from pathlib import Path
import os
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedGroupKFold
from statsmodels.miscmodels.ordinal_model import OrderedModel

from models.common import load_yaml, ordinal_metrics, patient_groups, save_json
from preprocessing.feature_encoding import (
    build_preprocessor,
    eligible_rows,
    select_features,
    transformed_feature_names,
)


def expand_probabilities(probabilities, classes):
    probs = np.asarray(probabilities, dtype=float)
    out = np.zeros((probs.shape[0], 7), dtype=float)
    for position, label in enumerate(classes):
        out[:, int(label)] = probs[:, position]
    row_sum = out.sum(axis=1)
    if np.any(row_sum <= 0):
        raise RuntimeError("Ordinal model produced an empty probability row.")
    return out / row_sum[:, None]


def fit_ordered_logit(X, y):
    settings = load_yaml("config/hyperparameters.yaml")["ordered_logit"]
    model = OrderedModel(y.astype(int), X, distr="logit")
    result = model.fit(
        method=str(settings["method"]),
        maxiter=int(settings["maxiter"]),
        disp=bool(settings["disp"]),
    )
    return result


def main():
    df = pd.read_csv("data/features_development.csv", low_memory=False)
    mask = eligible_rows(df, "pcoma_e_ordinal", "mrs_90d")
    eligible = df.loc[mask].copy()
    X = select_features(eligible, "pcoma_e_ordinal")
    y = pd.to_numeric(eligible["mrs_90d"], errors="raise").astype(int)
    if not y.between(0, 6).all():
        raise ValueError("mrs_90d must be between 0 and 6.")
    groups = patient_groups(eligible)

    seeds = load_yaml("config/random_seeds.yaml")
    splitter = StratifiedGroupKFold(
        n_splits=int(seeds["outer_cv_splits"]),
        shuffle=True,
        random_state=int(seeds["global_seed"]),
    )
    oof = np.full((len(eligible), 7), np.nan, dtype=float)
    fold_ids = np.full(len(eligible), -1, dtype=int)
    convergence = []

    for fold, (train_idx, validation_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        preprocessor = build_preprocessor(
            X.iloc[train_idx], scale_numeric=True, drop_first=True
        )
        X_train = preprocessor.fit_transform(X.iloc[train_idx])
        X_validation = preprocessor.transform(X.iloc[validation_idx])
        selector = VarianceThreshold(0.0)
        X_train = selector.fit_transform(X_train)
        X_validation = selector.transform(X_validation)
        classes = np.sort(y.iloc[train_idx].unique())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_ordered_logit(X_train, y.iloc[train_idx].to_numpy())
        predicted = result.model.predict(result.params, exog=X_validation)
        oof[validation_idx] = expand_probabilities(predicted, classes)
        fold_ids[validation_idx] = fold
        converged = bool(result.mle_retvals.get("converged", True))
        convergence.append(converged)
        print(f"PComA-E ordinal fold {fold}: converged={converged}")

    if np.isnan(oof).any():
        raise RuntimeError("PComA-E ordinal OOF probabilities are incomplete.")
    metrics = ordinal_metrics(y, oof)
    metrics["fold_convergence"] = convergence
    metrics["grouped_by"] = "study_patient_id"

    final_preprocessor = build_preprocessor(X, scale_numeric=True, drop_first=True)
    X_full = final_preprocessor.fit_transform(X)
    final_selector = VarianceThreshold(0.0)
    X_full = final_selector.fit_transform(X_full)
    final_classes = np.sort(y.unique())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_result = fit_ordered_logit(X_full, y.to_numpy())

    output = Path("models/pcoma_e/ordinal")
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "preprocessor": final_preprocessor,
        "selector": final_selector,
        "result": final_result,
        "classes": final_classes,
        "raw_features": list(X.columns),
        "transformed_features": transformed_feature_names(final_preprocessor),
    }
    joblib.dump(bundle, output / "pcoma_e_ordinal_model.joblib")
    np.save(output / "oof_probabilities.npy", oof)
    np.save(output / "oof_labels.npy", y.to_numpy())
    np.save(output / "oof_fold.npy", fold_ids)

    id_columns = [column for column in [
        "study_patient_id", "study_case_id", "aneurysm_id", "procedure_id", "index_year"
    ] if column in eligible.columns]
    predictions = eligible[id_columns].copy()
    predictions["observed_mrs"] = y.to_numpy()
    predictions["fold"] = fold_ids
    for category in range(7):
        predictions[f"prob_mrs_{category}"] = oof[:, category]
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    save_json(output / "metrics.json", metrics)
    print(f"PComA-E ordinal OOF metrics: {metrics}")


if __name__ == "__main__":
    main()
