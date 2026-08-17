from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

MISSINGNESS_CODES = {"NA", "NR", "ND", "UA", "UNK", "UTA", "IND", "SRC", "LTFU"}
BINARY_COLUMNS = {
    "index_case_flag", "pcoa_incorporated_neck", "pcoa_adherent_to_sac",
    "perforator_origin_from_p1", "pcoa_stretched_over_dome", "cn3_grooving_present",
    "preop_cn3_palsy", "preop_cn3_pupil_involvement", "pcoma_c_exposure_component",
    "pcoma_c_reconstruction_component", "flow_rescue_manoeuvre_flag",
    "rupture_escalation_flag", "derived_high_complexity_case",
    "new_postop_dwi_lesion_flag", "new_postop_ct_hypodensity_flag",
    "new_postop_ischaemic_deficit_flag", "lesion_anatomically_concordant_flag",
    "preexisting_infarct_flag", "delayed_cerebral_ischaemia_flag",
    "vasospasm_related_infarct_flag", "venous_infarction_flag",
    "nonprocedural_embolic_lesion_flag", "ischaemia_mechanism_indeterminate_flag",
    "derived_procedure_related_ischemia_flag", "cn3_complete_recovery_event",
    "cn3_complete_recovery_90d", "cn3_complete_recovery_6m",
    "cn3_complete_recovery_12m", "cn3_complete_recovery_24m",
    "death_before_cn3_recovery", "mrs_90d_window_eligible_flag",
    "derived_poor_outcome_90d",
}
NUMERIC_COLUMNS = {
    "index_year", "age_years", "aneurysm_max_diameter_mm", "height_mm",
    "neck_width_mm", "dome_to_neck_ratio", "aspect_ratio", "pcoa_diameter_mm",
    "p1_diameter_mm", "acha_distance_from_neck_mm", "perforator_count_estimated",
    "largest_perforator_diameter_mm", "parent_ica_curvature_deg",
    "parent_ica_tortuosity_index", "distance_to_pcoa_ostium_mm",
    "preop_cn3_duration_days", "temporary_clip_cumulative_sec",
    "clip_reposition_count", "postop_imaging_time_hours", "time_to_first_improvement_days",
    "time_to_complete_recovery_days", "cn3_last_assessment_days", "death_time_days",
    "mrs_90d", "mrs_90d_assessment_day", "baseline_mrs", "preop_nihss_total",
    "admission_gcs_total", "hunt_hess_grade", "wfns_grade",
}


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def preserve_missingness_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in list(df.columns):
        raw = df[column].astype("string").str.strip()
        code_mask = raw.isin(MISSINGNESS_CODES)
        if code_mask.any():
            df[f"{column}__missing_code"] = raw.where(code_mask, pd.NA)
            df.loc[code_mask, column] = np.nan
        blank_mask = raw.eq("")
        if blank_mask.any():
            df.loc[blank_mask, column] = np.nan
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = preserve_missingness_codes(df)
    for column in NUMERIC_COLUMNS & set(df.columns):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in BINARY_COLUMNS & set(df.columns):
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.dropna().loc[~values.dropna().isin([0, 1])]
        if not invalid.empty:
            raise ValueError(f"{column} contains values outside 0/1.")
        df[column] = values.astype("Int64")
    for column in df.select_dtypes(include=["object", "string"]).columns:
        if column.endswith("__missing_code"):
            continue
        df[column] = df[column].astype("string").str.strip().replace("", pd.NA)
    return df


def main():
    paths = load_yaml("config/paths.yaml")
    source = Path(paths["raw_data"])
    if not source.exists():
        raise FileNotFoundError(
            f"Approved de-identified dataset not found at {source}. No synthetic fallback is permitted."
        )
    df = pd.read_csv(source, low_memory=False, keep_default_na=False)
    df = clean_columns(df)
    destination = Path(paths["cleaned_data"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    print(f"Cleaning complete: {destination} ({len(df)} rows, {len(df.columns)} columns).")


if __name__ == "__main__":
    main()
