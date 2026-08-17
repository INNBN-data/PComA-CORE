from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_model_settings(path: str = "config/model_settings.yaml"):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_model_spec(model_key: str) -> dict:
    settings = load_model_settings()
    if model_key not in settings:
        raise KeyError(f"Unknown model key: {model_key}")
    return settings[model_key]


def get_raw_features(model_key: str) -> list[str]:
    return list(get_model_spec(model_key)["features"])


def select_features(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    features = get_raw_features(model_key)
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise KeyError(f"Missing required {model_key} features: {missing}")
    return df.loc[:, features].copy()


def eligible_rows(df: pd.DataFrame, model_key: str, target: str | None = None) -> pd.Series:
    spec = get_model_spec(model_key)
    mask = pd.Series(True, index=df.index)
    eligibility_column = spec.get("eligibility_column")
    if eligibility_column:
        if eligibility_column not in df.columns:
            raise KeyError(f"Missing eligibility column: {eligibility_column}")
        mask &= pd.to_numeric(df[eligibility_column], errors="coerce").eq(1)
    if target:
        if target not in df.columns:
            raise KeyError(f"Missing target column: {target}")
        mask &= df[target].notna()
    return mask


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = False, drop_first: bool = False):
    numeric_columns = X.select_dtypes(include=[np.number, "boolean"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    transformers = []
    if numeric_columns:
        numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        drop="first" if drop_first else None,
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("No predictor columns were available for preprocessing.")
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def transformed_feature_names(preprocessor) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def save_feature_names(model_key: str, columns, root: str = "models"):
    path = Path(root) / model_key
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "feature_names.json", "w", encoding="utf-8") as handle:
        json.dump(list(columns), handle, ensure_ascii=False, indent=2)
