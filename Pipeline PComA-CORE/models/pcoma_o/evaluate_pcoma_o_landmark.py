from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd
from preprocessing.feature_encoding import eligible_rows


def main():
    df = pd.read_csv("data/features_temporal.csv", low_memory=False)
    mask = eligible_rows(df, "pcoma_o_landmark", "derived_cn3_nonrecovery_12m")
    eligible = df.loc[mask]
    non_recovery = int((pd.to_numeric(eligible["derived_cn3_nonrecovery_12m"]) == 1).sum())
    print(
        f"PComA-O temporal stress-test cohort: n={len(eligible)}, persistent-dysfunction events={non_recovery}. "
        "The manuscript does not treat this sparse cohort as sufficient for a standalone temporal AUROC."
    )


if __name__ == "__main__":
    main()
