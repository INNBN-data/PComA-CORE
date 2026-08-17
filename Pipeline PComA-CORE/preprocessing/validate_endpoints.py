from __future__ import annotations

from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd
import yaml

from preprocessing.endpoint_definitions import (
    EndpointDefinitionError,
    compute_pcoma_c_endpoint,
    compute_pcoma_e_endpoints,
    compute_pcoma_o_survival_fields,
    compute_pcoma_r_endpoint,
)


def load_paths():
    with open("config/paths.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def audit(path: str):
    df = pd.read_csv(path, low_memory=False)
    report = {"file": path, "rows": int(len(df)), "status": {}}
    functions = {
        "pcoma_c": compute_pcoma_c_endpoint,
        "pcoma_r": compute_pcoma_r_endpoint,
        "pcoma_o": compute_pcoma_o_survival_fields,
        "pcoma_e": compute_pcoma_e_endpoints,
    }
    for name, function in functions.items():
        try:
            derived = function(df)
            report["status"][name] = {
                "ok": True,
                "columns": list(derived.columns),
                "nonmissing": {column: int(derived[column].notna().sum()) for column in derived.columns},
            }
        except EndpointDefinitionError as exc:
            report["status"][name] = {"ok": False, "reason": str(exc)}
    return report


def main():
    paths = load_paths()
    reports = [
        audit(paths["features_development"]),
        audit(paths["features_temporal"]),
    ]
    destination = Path("results/generated/endpoint_validation.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    if not all(item["ok"] for report in reports for item in report["status"].values()):
        raise SystemExit("One or more endpoints failed strict validation.")


if __name__ == "__main__":
    main()
