from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from models.common import load_yaml, patient_groups, save_json
from preprocessing.feature_encoding import build_preprocessor, eligible_rows, select_features


def cluster_bootstrap_indices(groups, rng):
    unique_groups = np.asarray(pd.unique(groups))
    sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
    return np.concatenate([np.flatnonzero(np.asarray(groups) == group) for group in sampled])


def fit_predict(X_train, y_train, X_test):
    preprocessor = build_preprocessor(X_train, scale_numeric=False)
    Xt = preprocessor.fit_transform(X_train)
    Xv = preprocessor.transform(X_test)
    model = CatBoostClassifier(**load_yaml("config/hyperparameters.yaml")["catboost"])
    model.fit(Xt, y_train)
    return model.predict_proba(Xt)[:, 1], model.predict_proba(Xv)[:, 1]


def main():
    df = pd.read_csv("data/features_development.csv", low_memory=False)
    mask = eligible_rows(df, "pcoma_c", "derived_high_complexity_case")
    eligible = df.loc[mask].copy()
    X = select_features(eligible, "pcoma_c")
    y = pd.to_numeric(eligible["derived_high_complexity_case"], errors="raise").astype(int).to_numpy()
    groups = patient_groups(eligible).to_numpy()

    settings = load_yaml("config/random_seeds.yaml")
    rng = np.random.default_rng(int(settings["global_seed"]))
    n_bootstrap = int(settings["n_bootstrap"])

    apparent_train, apparent_test = fit_predict(X, y, X)
    apparent_auc = roc_auc_score(y, apparent_test)
    optimism = []
    test_auc = []
    attempts = 0
    while len(optimism) < n_bootstrap and attempts < n_bootstrap * 10:
        attempts += 1
        idx = cluster_bootstrap_indices(groups, rng)
        y_boot = y[idx]
        if np.unique(y_boot).size < 2:
            continue
        boot_pred, original_pred = fit_predict(X.iloc[idx], y_boot, X)
        boot_auc = roc_auc_score(y_boot, boot_pred)
        original_auc = roc_auc_score(y, original_pred)
        optimism.append(float(boot_auc - original_auc))
        test_auc.append(float(original_auc))
        if len(optimism) % 100 == 0:
            print(f"Completed {len(optimism)}/{n_bootstrap} valid patient-cluster bootstraps.")

    if len(optimism) != n_bootstrap:
        raise RuntimeError("Unable to obtain the requested valid bootstrap replicates.")
    mean_optimism = float(np.mean(optimism))
    output = {
        "replicates": n_bootstrap,
        "apparent_auc": float(apparent_auc),
        "mean_optimism": mean_optimism,
        "optimism_corrected_auc": float(apparent_auc - mean_optimism),
        "test_auc_standard_error": float(np.std(test_auc, ddof=1)),
        "bootstrap_unit": "study_patient_id",
    }
    save_json("validation/generated/bootstrap_optimism_pcoma_c.json", output)
    print(output)


if __name__ == "__main__":
    main()
