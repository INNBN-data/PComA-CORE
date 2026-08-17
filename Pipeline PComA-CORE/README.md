# PComA-CORE: Manuscript-Concordant Reproducibility Repository

This repository accompanies the manuscript:

> **Posterior Communicating Artery Aneurysm Microsurgery: PComA-CORE, an Anatomy-Informed Explainable AI Framework for Complexity, Neurovascular Risk, Oculomotor Recovery and Functional Outcome.**

PComA-CORE was developed from a **real, fully de-identified clinical cohort** treated at the National Institute of Neurological Disorders and Neurovascular Disease (INNBN), Bucharest, Romania.

## Repository status

This public archive is a **manuscript-concordant reference implementation**. It contains code, schemas, endpoint rules, documentation, locked manuscript-reported metrics, and the author-supplied manuscript figures. It does not contain patient-level data, reversible linkage keys, operator identifiers, or free-text clinical records.

Exact numerical reproduction of the published models requires the locked de-identified analytical dataset, the original resampling assignments, and the final trained model and calibration objects. The values in `results/reported_metrics.yaml` are manuscript targets and are never overwritten by newly executed analyses.

## Data statement

- The source cohort is real and fully de-identified.
- No synthetic or simulated patient rows are included.
- No synthetic fallback is permitted by the code.
- The approved analytical dataset is expected at `data/private/pcoma_registry.csv` and remains Git-ignored.
- External patient-level access requires institutional approval, applicable ethics review, and an appropriate data-use agreement.

## Repository structure

- `config/` — fixed periods, model specifications, hyperparameters, paths, and seeds.
- `data/` — reference data dictionary, missingness codes, schema-only CSV, and structural validation.
- `preprocessing/` — cleaning, fixed temporal gating, endpoint validation/derivation, feature engineering, and fold-safe preprocessing utilities.
- `models/` — PComA-C, PComA-R, PComA-O survival and 12-month landmark models, and PComA-E ordinal and binary models.
- `validation/` — patient-grouped nested validation, bootstrap optimism, learning curves, permutation testing, and temporal evaluation.
- `calibration/` — development-derived recalibration and decision-curve analysis.
- `explainability/` — SHAP, ALE, permutation importance, and feature-stability analyses.
- `uncertainty/` — cross-fitted conformal prediction-set analysis.
- `score_distillation/` — manuscript-concordant sparse instruments where locked integer weights are available.
- `figures/` — author-supplied manuscript figures and scripts for generated diagnostic plots.
- `docs/` — study metadata, ethics, governance, reproducibility, figure provenance, and release notes.
- `results/` — locked manuscript-reported values and generated analysis outputs.

## Core safeguards

1. Development period: **1997–2020**.
2. Temporal validation period: **2021–2026**.
3. All resampling is grouped by `study_patient_id`.
4. No temporal-validation observation is used to learn imputation, scaling, encoding, feature selection, tuning, thresholding, model combination, or calibration.
5. Missing outcome values remain missing; they are never converted to absence of an event.
6. Endpoint derivation is strict and fails when required source variables are absent or disagree with a supplied derived endpoint.
7. One row represents one aneurysm treatment episode; repeated observations from one patient remain linked through de-identified study identifiers.

## Installation

Python 3.9 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

R analyses use the packages documented in `R_session_info.txt`.

## Reference workflow

```bash
python data/validate_dataset.py --strict-manuscript
python preprocessing/data_cleaning.py
python preprocessing/temporal_gating.py
python preprocessing/feature_engineering.py
python preprocessing/missing_data.py

python models/pcoma_c/train_pcoma_c.py
python models/pcoma_r/train_pcoma_r.py
python models/pcoma_o/train_pcoma_o_aft.py
python models/pcoma_o/train_pcoma_o_landmark.py
python models/pcoma_e/train_pcoma_e_binary.py
python models/pcoma_e/train_pcoma_e_ordinal.py

python validation/temporal_evaluation.py
python calibration/calibration_analysis.py
python uncertainty/conformal_prediction.py
python figures/manuscript_figures.py
```

The convenience command `python run_pipeline.py` executes the principal reference workflow after checking that the approved private dataset is present.

## Important score note

The manuscript describes PComA-R as a 0–17-point instrument, but the supplied reference script did not contain a locked set of integer weights summing to 17. The repository therefore does not invent those weights. See `docs/CONCORDANCE_NOTES.md` and `score_distillation/pcoma_r_score.py`.

## Citation and license

Use `CITATION.cff`. Code and documentation are released under the MIT License. The license does not grant access rights to clinical data.

## Clinical disclaimer

PComA-CORE remains a derivation-stage research framework. It is not an autonomous clinical decision system and requires independent multicentre and prospective validation before clinical deployment.
