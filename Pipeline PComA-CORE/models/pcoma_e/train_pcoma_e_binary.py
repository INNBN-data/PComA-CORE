from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd
from sklearn.linear_model import LogisticRegression

from models.binary_training import train_grouped_binary
from models.common import load_yaml


def main():
    df = pd.read_csv("data/features_development.csv", low_memory=False)
    params = load_yaml("config/hyperparameters.yaml")["elastic_net"]
    estimator = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=1.0 / float(params["alpha"]),
        l1_ratio=float(params["l1_ratio"]),
        max_iter=int(params["max_iter"]),
        random_state=2024,
    )
    metrics = train_grouped_binary(
        df=df,
        model_key="pcoma_e_binary",
        target="derived_poor_outcome_90d",
        estimator=estimator,
        output_dir="models/pcoma_e",
        scale_numeric=True,
    )
    print(f"PComA-E binary OOF metrics: {metrics}")


if __name__ == "__main__":
    main()
