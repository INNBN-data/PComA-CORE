from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve


def main():
    labels_path = Path("models/pcoma_c/oof_labels.npy")
    probabilities_path = Path("models/pcoma_c/oof_proba.npy")
    if not labels_path.exists() or not probabilities_path.exists():
        raise FileNotFoundError("Train PComA-C before generating diagnostic figures.")
    y = np.load(labels_path)
    p = np.load(probabilities_path)
    destination = Path("figures/generated")
    destination.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"OOF AUC = {roc_auc_score(y, p):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("PComA-C reference OOF ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination / "pcoma_c_oof_roc.png", dpi=300)
    plt.close()

    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(predicted, observed, marker="o", label="PComA-C")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event fraction")
    plt.title("PComA-C reference OOF calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination / "pcoma_c_oof_calibration.png", dpi=300)
    plt.close()
    print("Generated reference diagnostic figures. Fixed manuscript figures remain in figures/manuscript/.")


if __name__ == "__main__":
    main()
