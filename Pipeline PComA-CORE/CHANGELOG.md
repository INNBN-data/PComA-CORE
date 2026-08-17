# Changelog

## 1.0.0 — 2026-08-17

- Created a real-cohort, patient-level-data-free public reference repository.
- Removed synthetic dataset generation and all synthetic fallback behaviour.
- Preserved the manuscript as the authoritative source for periods, endpoints, terminology, thresholds, and reported results.
- Added exact author order, affiliations, ethics wording, consent-waiver wording, and official institutional name.
- Added canonical de-identified identifiers and patient-grouped resampling safeguards.
- Moved all learned missing-data processing into development-only or fold-local model pipelines.
- Added strict endpoint validation and eliminated silent zero defaults and swallowed exceptions.
- Separated PComA-O XGBoost-AFT survival modelling from the 12-month XGBoost landmark model.
- Separated PComA-E ordinal modelling from the secondary elastic-net binary model.
- Added author-supplied manuscript figures, figure-provenance documentation, structural tests, and cryptographic manifests.
- Recorded the unresolved PComA-R integer-weight discrepancy without fabricating weights.
