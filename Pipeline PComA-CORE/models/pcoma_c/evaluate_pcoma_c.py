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
        model_key="pcoma_c",
        target="derived_high_complexity_case",
        model_path="models/pcoma_c/pcoma_c_model.joblib",
        output_dir="models/pcoma_c",
    )
    print(f"PComA-C temporal metrics: {metrics}")


if __name__ == "__main__":
    main()
