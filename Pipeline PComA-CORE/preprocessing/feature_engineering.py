from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from preprocessing.endpoint_definitions import (
    compute_pcoma_c_endpoint,
    compute_pcoma_e_endpoints,
    compute_pcoma_o_survival_fields,
    compute_pcoma_r_endpoint,
)


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "aspect_ratio" not in df.columns:
        df["aspect_ratio"] = np.nan
    if {"height_mm", "neck_width_mm"}.issubset(df.columns):
        mask = df["aspect_ratio"].isna() & df["height_mm"].notna() & df["neck_width_mm"].gt(0)
        df.loc[mask, "aspect_ratio"] = df.loc[mask, "height_mm"] / df.loc[mask, "neck_width_mm"]
    return df


def add_or_validate_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for derived in [
        compute_pcoma_c_endpoint(df),
        compute_pcoma_r_endpoint(df),
        compute_pcoma_o_survival_fields(df),
        compute_pcoma_e_endpoints(df),
    ]:
        for column in derived.columns:
            df[column] = derived[column]
    return df


def process(source: str, destination: str):
    df = pd.read_csv(source, low_memory=False)
    df = add_derived_features(df)
    df = add_or_validate_endpoints(df)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    print(f"Feature and endpoint processing complete: {destination} ({len(df)} rows).")


def main():
    paths = load_yaml("config/paths.yaml")
    process(paths["development_data"], paths["features_development"])
    process(paths["temporal_data"], paths["features_temporal"])


if __name__ == "__main__":
    main()
