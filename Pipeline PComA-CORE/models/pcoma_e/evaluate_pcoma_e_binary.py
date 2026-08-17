from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd
from models.binary_training import evaluate_grouped_binary


def main():
    df = pd.read_csv("data/features_temporal.csv", low_memory=False)
    metrics = evaluate_grouped_binary(
        df=df,
        model_key="pcoma_e_binary",
        target="derived_poor_outcome_90d",
        model_path="models/pcoma_e/pcoma_e_binary_model.joblib",
        output_dir="models/pcoma_e",
    )
    print(f"PComA-E binary temporal metrics: {metrics}")


if __name__ == "__main__":
    main()
