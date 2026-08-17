from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd
from catboost import CatBoostClassifier

from models.binary_training import train_grouped_binary
from models.common import load_yaml


def main():
    df = pd.read_csv("data/features_development.csv", low_memory=False)
    params = load_yaml("config/hyperparameters.yaml")["catboost"]
    estimator = CatBoostClassifier(**params)
    metrics = train_grouped_binary(
        df=df,
        model_key="pcoma_r",
        target="derived_procedure_related_ischemia_flag",
        estimator=estimator,
        output_dir="models/pcoma_r",
        scale_numeric=False,
    )
    print(f"PComA-R OOF metrics: {metrics}")


if __name__ == "__main__":
    main()
