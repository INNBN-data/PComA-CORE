# Analysis Pipeline

1. Validate the incoming de-identified dataset.
2. Clean values while preserving explicit missingness reasons.
3. Apply the fixed temporal split before any learned preprocessing.
4. Derive or validate endpoints separately in development and temporal data.
5. Perform model-specific imputation, encoding, and scaling within each training fold.
6. Produce patient-grouped out-of-fold predictions in development data.
7. Fit the final model on the complete development cohort only.
8. Apply the locked model and development-derived threshold/calibration to the untouched 2021–2026 temporal cohort.
9. Assess discrimination, calibration, clinical net benefit, explainability, uncertainty, score distillation, and stability.
10. Compare generated values against `results/reported_metrics.yaml` without overwriting the manuscript targets.

PComA-O includes two separate analyses: XGBoost accelerated failure-time modelling for longitudinal recovery and a fixed 12-month XGBoost landmark classifier, with persistent CN III dysfunction as the positive class. PComA-E includes a primary ordinal outcome model and a secondary elastic-net binary model for mRS >2.
