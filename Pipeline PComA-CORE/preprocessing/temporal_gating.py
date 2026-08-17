from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def split_temporal(df: pd.DataFrame, settings: dict):
    if "index_year" not in df.columns:
        raise KeyError("index_year is required. Calendar years are never fabricated by this code.")
    years = pd.to_numeric(df["index_year"], errors="raise")
    development = df.loc[years.between(
        int(settings["development_start_year"]),
        int(settings["development_end_year"]),
    )].copy()
    temporal = df.loc[years.between(
        int(settings["temporal_start_year"]),
        int(settings["temporal_end_year"]),
    )].copy()
    if len(development) + len(temporal) != len(df):
        raise ValueError("One or more rows fall outside the fixed 1997-2026 study periods.")

    overlap = set(development["study_patient_id"].astype(str)) & set(
        temporal["study_patient_id"].astype(str)
    )
    if overlap:
        raise ValueError(
            f"{len(overlap)} patients cross the development and temporal cohorts. "
            "Resolve the locked cohort definition before analysis."
        )

    if bool(settings.get("enforce_manuscript_counts", False)):
        expected = (
            int(settings["expected_development_n"]),
            int(settings["expected_temporal_n"]),
        )
        observed = (len(development), len(temporal))
        if observed != expected:
            raise ValueError(f"Temporal split counts differ: observed={observed}, expected={expected}.")
    return development, temporal


def main():
    paths = load_yaml("config/paths.yaml")
    settings = load_yaml("config/model_settings.yaml")["general"]
    source = Path(paths["cleaned_data"])
    if not source.exists():
        raise FileNotFoundError(f"Run preprocessing/data_cleaning.py first: {source}")
    df = pd.read_csv(source, low_memory=False)
    development, temporal = split_temporal(df, settings)
    Path(paths["development_data"]).parent.mkdir(parents=True, exist_ok=True)
    development.to_csv(paths["development_data"], index=False)
    temporal.to_csv(paths["temporal_data"], index=False)
    print(f"Development: {len(development)}; temporal: {len(temporal)}.")


if __name__ == "__main__":
    main()
