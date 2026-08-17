from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd
import yaml

MISSINGNESS_CODES = {"NA", "NR", "ND", "UA", "UNK", "UTA", "IND", "SRC", "LTFU"}
REQUIRED_IDENTIFIERS = [
    "study_patient_id",
    "study_case_id",
    "aneurysm_id",
    "procedure_id",
    "linked_original_case_id",
    "index_case_flag",
    "index_year",
]
DIRECT_IDENTIFIER_PATTERNS = [
    r"(^|_)name($|_)", r"(^|_)surname($|_)", r"cnp", r"ssn", r"passport",
    r"phone", r"telephone", r"email", r"address", r"medical_record",
    r"hospital_number", r"date_of_birth", r"dob", r"free_text", r"operator_name",
    r"surgeon_name",
]


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate an incoming PComA-CORE analytical CSV.")
    parser.add_argument("--data", default=None, help="CSV path; defaults to config/paths.yaml raw_data.")
    parser.add_argument(
        "--strict-manuscript",
        action="store_true",
        help="Require the manuscript totals 687/564/123 and fixed study years.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_yaml("config/paths.yaml")
    settings = load_yaml("config/model_settings.yaml")["general"]
    source = Path(args.data or paths["raw_data"])
    if not source.exists():
        print(f"ERROR: dataset not found: {source}")
        return 1

    df = pd.read_csv(source, low_memory=False, keep_default_na=False)
    errors: list[str] = []
    warnings: list[str] = []

    missing_ids = [column for column in REQUIRED_IDENTIFIERS if column not in df.columns]
    if missing_ids:
        errors.append(f"Missing required identifier/time columns: {missing_ids}")

    suspicious = []
    for column in df.columns:
        low = column.lower()
        if any(re.search(pattern, low) for pattern in DIRECT_IDENTIFIER_PATTERNS):
            suspicious.append(column)
    if suspicious:
        errors.append(f"Potential direct-identifier columns detected: {suspicious}")

    for column in ["study_patient_id", "study_case_id", "aneurysm_id", "procedure_id"]:
        if column in df.columns and (df[column].astype(str).str.strip() == "").any():
            errors.append(f"{column} contains empty values.")

    if "study_case_id" in df.columns and df["study_case_id"].duplicated().any():
        errors.append("study_case_id must uniquely identify one aneurysm treatment episode.")

    if "index_case_flag" in df.columns:
        values = pd.to_numeric(df["index_case_flag"], errors="coerce")
        if values.isna().any() or not set(values.unique()).issubset({0, 1}):
            errors.append("index_case_flag must contain only 0 or 1.")

    if "index_year" in df.columns:
        years = pd.to_numeric(df["index_year"], errors="coerce")
        if years.isna().any():
            errors.append("index_year contains missing or non-numeric values.")
        else:
            outside = ~years.between(
                int(settings["development_start_year"]),
                int(settings["temporal_end_year"]),
            )
            if outside.any():
                errors.append("index_year contains values outside 1997-2026.")

            development = years <= int(settings["development_end_year"])
            temporal = years >= int(settings["temporal_start_year"])
            if args.strict_manuscript:
                expected = {
                    "total": int(settings["expected_total_n"]),
                    "development": int(settings["expected_development_n"]),
                    "temporal": int(settings["expected_temporal_n"]),
                }
                observed = {
                    "total": len(df),
                    "development": int(development.sum()),
                    "temporal": int(temporal.sum()),
                }
                if observed != expected:
                    errors.append(f"Manuscript cohort counts differ: observed={observed}, expected={expected}")

            if "study_patient_id" in df.columns:
                dev_patients = set(df.loc[development, "study_patient_id"].astype(str))
                temp_patients = set(df.loc[temporal, "study_patient_id"].astype(str))
                overlap = sorted(dev_patients & temp_patients)
                if overlap:
                    errors.append(
                        f"{len(overlap)} patients cross development and temporal periods. "
                        "Resolve according to the locked cohort before analysis."
                    )

    known_columns = set(pd.read_csv("data/data_dictionary.csv")["variable"].astype(str))
    undocumented = sorted(set(df.columns) - known_columns)
    if undocumented:
        warnings.append(
            f"{len(undocumented)} columns are not present in the public reference dictionary. "
            "This may be expected for the full 779-variable registry, but the locked dictionary must document them."
        )

    for column in df.columns:
        values = set(df[column].astype(str).str.strip().unique())
        code_like = {value for value in values if value in MISSINGNESS_CODES}
        unknown_codes = {value for value in values if value.startswith("MISS_")}
        if unknown_codes:
            errors.append(f"{column} contains unsupported missingness codes: {sorted(unknown_codes)}")
        if code_like and column in REQUIRED_IDENTIFIERS[:-1]:
            errors.append(f"Identifier {column} contains missingness codes: {sorted(code_like)}")

    print(f"Dataset: {source}")
    print(f"Rows: {len(df)}; columns: {len(df.columns)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    if errors:
        return 1
    print("Dataset structural validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
