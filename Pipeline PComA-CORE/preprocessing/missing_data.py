"""Missingness reporting and fold-safe preprocessing policy.

No imputed cohort-wide CSV is created. Imputation, categorical encoding, and
scaling are fitted inside each training fold by the model scripts and are then
fitted once on the complete development cohort before temporal application.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def report(source: str, destination: str):
    df = pd.read_csv(source, low_memory=False)
    rows = []
    for column in df.columns:
        if column.endswith("__missing_code"):
            continue
        rows.append(
            {
                "variable": column,
                "n": len(df),
                "missing_n": int(df[column].isna().sum()),
                "missing_fraction": float(df[column].isna().mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["missing_fraction", "variable"], ascending=[False, True])
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Missingness report: {path}")


def main():
    paths = load_yaml("config/paths.yaml")
    report(paths["features_development"], paths["missingness_development"])
    report(paths["features_temporal"], paths["missingness_temporal"])
    print("No outcome or predictor was imputed outside a model-training fold.")


if __name__ == "__main__":
    main()
