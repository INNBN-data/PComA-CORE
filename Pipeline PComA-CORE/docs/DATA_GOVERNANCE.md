# Data Governance

## Provenance

The PComA-CORE cohort consists entirely of real, fully de-identified clinical observations from the National Institute of Neurological Disorders and Neurovascular Disease.

## Public archive

The public archive contains no patient-level rows. It includes only code, schemas, dictionaries, manuscript-reported aggregate results, and author-supplied figures.

## Required analytical identifiers

- `study_patient_id`
- `study_case_id`
- `aneurysm_id`
- `procedure_id`
- `linked_original_case_id`
- `index_case_flag`

All resampling must be grouped by `study_patient_id`. Operator identifiers and reversible linkage variables remain in the secure institutional environment and are excluded from public outputs.

## Missingness codes

The accepted codes are `NA`, `NR`, `ND`, `UA`, `UNK`, `UTA`, `IND`, `SRC`, and `LTFU`. They are preserved in companion `__missing_code` fields during cleaning and are never interpreted automatically as zero.

## Access

Real-data access requires institutional approval, ethics review, and a data-use agreement. The MIT license covers code and documentation only; it does not grant access to clinical data.
