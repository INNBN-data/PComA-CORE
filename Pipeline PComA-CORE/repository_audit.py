from __future__ import annotations

from pathlib import Path
import ast
import csv
import hashlib
import json
import sys

import yaml

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "README.md", "LICENSE", "CITATION.cff", "VERSION",
    "config/model_settings.yaml", "config/hyperparameters.yaml",
    "data/data_dictionary.csv", "data/missingness_codes.csv",
    "data/pcoma_registry_schema.csv", "preprocessing/endpoint_definitions.py",
    "models/pcoma_c/train_pcoma_c.py", "models/pcoma_r/train_pcoma_r.py",
    "models/pcoma_o/train_pcoma_o_aft.py", "models/pcoma_o/train_pcoma_o_landmark.py",
    "models/pcoma_e/train_pcoma_e_binary.py", "models/pcoma_e/train_pcoma_e_ordinal.py",
    "results/reported_metrics.yaml", "docs/CONCORDANCE_NOTES.md",
    "figures/manuscript/pcoma_c_predictive_performance.png",
    "figures/manuscript/pcoma_r_anatomical_determinants.png",
    "figures/manuscript/pcoma_o_sequential_predictor_domains.png",
    "figures/manuscript/pcoma_core_learning_curves_bootstrap.png",
    "figures/manuscript/temporal_evolution_outcomes.png",
]
FORBIDDEN_FILES = [
    "data/generate_example_dataset.py", "data/example_dataset.csv", "data/synthetic_example.csv"
]
FORBIDDEN_PLACEHOLDERS = ["your-username", "10.xxxx", "Journal Name"]
EXPECTED_AUTHORS = ["Șerban", "Toader", "Ciurea", "Dănăilă", "Covache-Busuioc"]
EXPECTED_MISSINGNESS = ["NA", "NR", "ND", "UA", "UNK", "UTA", "IND", "SRC", "LTFU"]
EXPECTED_IDS = [
    "study_patient_id", "study_case_id", "aneurysm_id", "procedure_id",
    "linked_original_case_id", "index_case_flag",
]


def main() -> int:
    errors = []
    for relative in REQUIRED:
        if not (ROOT / relative).exists():
            errors.append(f"Missing required path: {relative}")
    for relative in FORBIDDEN_FILES:
        if (ROOT / relative).exists():
            errors.append(f"Forbidden synthetic/example artefact: {relative}")

    for path in sorted(ROOT.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"Syntax error in {path.relative_to(ROOT)}: {exc}")

    for relative in [
        "CITATION.cff", "config/model_settings.yaml", "config/hyperparameters.yaml",
        "config/random_seeds.yaml", "config/paths.yaml", "results/reported_metrics.yaml",
    ]:
        try:
            yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"YAML/CFF parse failure in {relative}: {exc}")

    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    authors = [item["family-names"] for item in citation.get("authors", [])]
    if authors != EXPECTED_AUTHORS:
        errors.append(f"CITATION author order differs: {authors}")

    text_paths = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in {
        ".md", ".txt", ".yaml", ".yml", ".cff", ".py", ".csv"
    }]
    for path in text_paths:
        if path.name == "repository_audit.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for placeholder in FORBIDDEN_PLACEHOLDERS:
            if placeholder in text:
                errors.append(f"Placeholder {placeholder!r} remains in {path.relative_to(ROOT)}")

    with (ROOT / "data/missingness_codes.csv").open(encoding="utf-8") as handle:
        codes = [row["code"] for row in csv.DictReader(handle)]
    if codes != EXPECTED_MISSINGNESS:
        errors.append(f"Missingness code order/content differs: {codes}")

    with (ROOT / "data/data_dictionary.csv").open(encoding="utf-8") as handle:
        dictionary_variables = [row["variable"] for row in csv.DictReader(handle)]
    missing_ids = [item for item in EXPECTED_IDS if item not in dictionary_variables]
    if missing_ids:
        errors.append(f"Data dictionary missing canonical identifiers: {missing_ids}")

    metrics = yaml.safe_load((ROOT / "results/reported_metrics.yaml").read_text(encoding="utf-8"))
    checks = {
        "cohort.total": metrics["cohort"]["total"] == 687,
        "cohort.development": metrics["cohort"]["development"] == 564,
        "cohort.temporal": metrics["cohort"]["temporal"] == 123,
        "pcoma_c.temporal_auc": metrics["pcoma_c"]["temporal_auc"] == 0.943,
        "pcoma_r.temporal_auc": metrics["pcoma_r"]["temporal_auc"] == 0.703,
        "pcoma_o.landmark_auc": metrics["pcoma_o"]["selected_landmark_auc"] == 0.902,
        "pcoma_e.temporal_auc": metrics["pcoma_e"]["binary_temporal_auc"] == 0.878,
    }
    for name, ok in checks.items():
        if not ok:
            errors.append(f"Locked metric check failed: {name}")

    result = {
        "required_paths_checked": len(REQUIRED),
        "python_files_parsed": len(list(ROOT.rglob("*.py"))),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
