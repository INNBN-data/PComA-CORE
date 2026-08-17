from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd
from xgboost import XGBClassifier

from models.binary_training import train_grouped_binary
from models.common import load_yaml


def main():
    df = pd.read_csv("data/features_development.csv", low_memory=False)
    params = load_yaml("config/hyperparameters.yaml")["xgboost_classifier"]
    estimator = XGBClassifier(**params)
    metrics = train_grouped_binary(
        df=df,
        model_key="pcoma_o_landmark",
        target="derived_cn3_nonrecovery_12m",
        estimator=estimator,
        output_dir="models/pcoma_o/landmark",
        scale_numeric=False,
    )
    print(f"PComA-O 12-month landmark OOF metrics: {metrics}")


if __name__ == "__main__":
    main()
