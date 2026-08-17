# Data Directory

This public directory contains no patient-level observations.

- `data_dictionary.csv` — reference bilingual dictionary for canonical variables used by the repository.
- `missingness_codes.csv` — accepted explicit missingness reasons.
- `pcoma_registry_schema.csv` — zero-row header template only.
- `validate_dataset.py` — incoming-data structural and manuscript-count validation.
- `private/` — Git-ignored location for the approved de-identified analytical CSV.
- `secure/` — documentation placeholder for variables that must remain in the institutional secure environment.

The complete 779-variable registry dictionary and locked analytical export should replace or extend this reference dictionary during controlled internal validation. No synthetic example dataset is included.
