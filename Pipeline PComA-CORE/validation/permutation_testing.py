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
from sklearn.model_selection import StratifiedGroupKFold

from models.common import load_yaml, patient_groups, save_json
from preprocessing.feature_encoding import build_preprocessor, eligible_rows, select_features


def oof_auc(X, y, fixed_splits):
    oof = np.full(len(y), np.nan, dtype=float)
    for train_idx, validation_idx in fixed_splits:
        preprocessor = build_preprocessor(X.iloc[train_idx], scale_numeric=False)
        Xt = preprocessor.fit_transform(X.iloc[train_idx])
        Xv = preprocessor.transform(X.iloc[validation_idx])
        model = CatBoostClassifier(**load_yaml("config/hyperparameters.yaml")["catboost"])
        model.fit(Xt, y[train_idx])
        oof[validation_idx] = model.predict_proba(Xv)[:, 1]
    return float(roc_auc_score(y, oof))


def main():
    df = pd.read_csv("data/features_development.csv", low_memory=False)
    mask = eligible_rows(df, "pcoma_c", "derived_high_complexity_case")
    eligible = df.loc[mask].copy()
    X = select_features(eligible, "pcoma_c")
    y = pd.to_numeric(eligible["derived_high_complexity_case"], errors="raise").astype(int).to_numpy()
    groups = patient_groups(eligible).to_numpy()

    settings = load_yaml("config/random_seeds.yaml")
    splitter = StratifiedGroupKFold(
        n_splits=int(settings["outer_cv_splits"]), shuffle=True,
        random_state=int(settings["global_seed"]),
    )
    fixed_splits = list(splitter.split(X, y, groups))
    baseline = oof_auc(X, y, fixed_splits)
    rng = np.random.default_rng(int(settings["global_seed"]))
    n_permutations = int(settings["n_permutations"])
    values = []
    for index in range(n_permutations):
        values.append(oof_auc(X, rng.permutation(y), fixed_splits))
        if (index + 1) % 50 == 0:
            print(f"Completed {index + 1}/{n_permutations} permutations.")
    p_value = (sum(value >= baseline for value in values) + 1) / (n_permutations + 1)
    output = {
        "baseline_grouped_oof_auc": baseline,
        "permutations": n_permutations,
        "p_value": float(p_value),
        "permuted_auc_mean": float(np.mean(values)),
        "permuted_auc_sd": float(np.std(values, ddof=1)),
        "note": "Fixed patient-grouped folds; labels are globally permuted and every model is refitted.",
    }
    save_json("validation/generated/permutation_test_pcoma_c.json", output)
    print(output)


if __name__ == "__main__":
    main()
