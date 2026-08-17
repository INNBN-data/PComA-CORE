import numpy as np
import pandas as pd

REQUIRED = [
    "preop_cn3_duration_days", "cn3_grooving_present",
    "preop_cn3_complete_or_partial", "neck_width_mm", "age_years", "pcoa_diameter_mm",
]


def compute_score(df):
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise KeyError(f"Missing PComA-O score variables: {missing}")
    score = np.zeros(len(df), dtype=int)
    score += (df["preop_cn3_duration_days"] >= 68.8).astype(int) * 6
    score += (df["cn3_grooving_present"] == 1).astype(int) * 5
    score += (df["preop_cn3_complete_or_partial"] == "Complete").astype(int) * 4
    score += (df["neck_width_mm"] >= 6.3).astype(int) * 3
    score += (df["age_years"] >= 58.2).astype(int) * 2
    score += (df["pcoa_diameter_mm"] <= 1.15).astype(int) * 2
    return score


def main():
    df = pd.read_csv("data/features_development.csv", low_memory=False)
    eligible = df.loc[df["preop_cn3_palsy"] == 1].copy()
    eligible["pcoma_o_score"] = compute_score(eligible)
    eligible.to_csv("data/development_with_pcoma_o_score.csv", index=False)


if __name__ == "__main__":
    main()
