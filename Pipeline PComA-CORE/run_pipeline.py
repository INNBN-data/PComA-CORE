from pathlib import Path
import subprocess
import sys

STEPS = [
    ["data/validate_dataset.py", "--strict-manuscript"],
    ["preprocessing/data_cleaning.py"],
    ["preprocessing/temporal_gating.py"],
    ["preprocessing/feature_engineering.py"],
    ["preprocessing/missing_data.py"],
    ["preprocessing/validate_endpoints.py"],
    ["models/pcoma_c/train_pcoma_c.py"],
    ["models/pcoma_r/train_pcoma_r.py"],
    ["models/pcoma_o/train_pcoma_o_aft.py"],
    ["models/pcoma_o/train_pcoma_o_landmark.py"],
    ["models/pcoma_e/train_pcoma_e_binary.py"],
    ["models/pcoma_e/train_pcoma_e_ordinal.py"],
    ["validation/temporal_evaluation.py"],
    ["calibration/calibration_analysis.py"],
    ["uncertainty/conformal_prediction.py"],
    ["score_distillation/pcoma_c_score.py"],
    ["score_distillation/pcoma_o_score.py"],
    ["score_distillation/pcoma_e_score.py"],
    ["figures/manuscript_figures.py"],
]


def main():
    data = Path("data/private/pcoma_registry.csv")
    if not data.exists():
        raise SystemExit(
            "The approved real de-identified dataset is missing at "
            "data/private/pcoma_registry.csv. No synthetic fallback will be used."
        )
    failures = []
    for arguments in STEPS:
        print("Running", " ".join(arguments), flush=True)
        result = subprocess.run([sys.executable, *arguments], check=False)
        if result.returncode != 0:
            failures.append(" ".join(arguments))
            break
    if failures:
        raise SystemExit(f"Pipeline stopped after failure: {failures[-1]}")
    print(
        "Reference pipeline completed. PComA-R sparse scoring remains intentionally disabled "
        "until the locked 0-17 integer weights are supplied."
    )


if __name__ == "__main__":
    main()
