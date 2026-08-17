import numpy as np
import pandas as pd

REQUIRED = [
    "rupture_status", "age_years", "preop_nihss_total", "baseline_mrs",
    "admission_gcs_total", "acha_distance_from_neck_mm",
]


def compute_score(df):
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise KeyError(f"Missing PComA-E score variables: {missing}")
    score = np.zeros(len(df), dtype=int)
    score += (df["rupture_status"] == "Ruptured").astype(int) * 6
    score += (df["age_years"] >= 59.7).astype(int) * 4
    score += (df["preop_nihss_total"] >= 4).astype(int) * 4
    score += (df["baseline_mrs"] >= 1).astype(int) * 3
    score += (df["admission_gcs_total"] <= 11).astype(int) * 3
    score += (df["acha_distance_from_neck_mm"] <= 3.11).astype(int) * 2
    return score


def main():
    df = pd.read_csv("data/features_temporal.csv", low_memory=False)
    eligible = df.loc[df["derived_poor_outcome_90d"].notna()].copy()
    eligible["pcoma_e_score"] = compute_score(eligible)
    eligible.to_csv("data/temporal_with_pcoma_e_score.csv", index=False)


if __name__ == "__main__":
    main()
