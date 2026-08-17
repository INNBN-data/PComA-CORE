import numpy as np
import pandas as pd

REQUIRED = [
    "neck_width_mm", "dome_to_neck_ratio", "pcoa_adherent_to_sac",
    "perforator_origin_from_p1", "parent_ica_curvature_deg", "pcoa_incorporated_neck",
]


def compute_score(df):
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise KeyError(f"Missing PComA-C score variables: {missing}")
    score = np.zeros(len(df), dtype=int)
    score += (df["neck_width_mm"] >= 4.5).astype(int) * 4
    score += (df["dome_to_neck_ratio"] <= 1.91).astype(int) * 3
    score += (df["pcoa_adherent_to_sac"] == 1).astype(int) * 3
    score += (df["perforator_origin_from_p1"] == 1).astype(int) * 3
    score += (df["parent_ica_curvature_deg"] >= 49.7).astype(int) * 3
    score += (df["pcoa_incorporated_neck"] == 1).astype(int) * 3
    return score


def main():
    df = pd.read_csv("data/features_temporal.csv", low_memory=False)
    df["pcoma_c_score"] = compute_score(df)
    df.to_csv("data/temporal_with_pcoma_c_score.csv", index=False)


if __name__ == "__main__":
    main()
