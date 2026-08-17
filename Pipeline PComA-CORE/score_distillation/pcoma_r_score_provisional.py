"""Traceability-only provisional weights from the supplied reference script.

Do not cite this 0-13 implementation as the manuscript's final 0-17 PComA-R score.
"""

import numpy as np


def compute_provisional_score(df):
    score = np.zeros(len(df), dtype=int)
    score += (df["aspect_ratio"] <= 2.08).astype(int) * 2
    score += (df["neck_width_mm"] >= 2.3).astype(int) * 2
    score += (df["pcoa_incorporated_neck"] == 1).astype(int) * 2
    score += (df["p1_diameter_mm"] <= 2.08).astype(int) * 2
    score += (df["pcoa_adherent_to_sac"] == 1).astype(int) * 2
    score += (df["largest_perforator_diameter_mm"] >= 1.2).astype(int) * 3
    return score
