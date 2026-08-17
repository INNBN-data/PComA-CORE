"""Strict manuscript-concordant endpoint definitions.

The functions never replace missing outcomes with zero and never silently accept
incomplete source-variable mappings. When both raw components and a supplied
derived endpoint are available, concordance is checked row by row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class EndpointDefinitionError(ValueError):
    """Raised when an endpoint cannot be derived or validated safely."""


def _require(df: pd.DataFrame, columns: list[str], endpoint: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise EndpointDefinitionError(
            f"{endpoint} requires source columns {missing}. "
            "Use the locked data dictionary rather than guessing or defaulting them to zero."
        )


def _binary(series: pd.Series, name: str, allow_missing: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if not allow_missing and values.isna().any():
        raise EndpointDefinitionError(f"{name} contains missing or non-numeric values.")
    invalid = values.dropna().loc[~values.dropna().isin([0, 1])]
    if not invalid.empty:
        raise EndpointDefinitionError(f"{name} contains values outside 0/1.")
    return values.astype("Int64")


def _validate_existing(existing: pd.Series, derived: pd.Series, name: str) -> None:
    existing_num = _binary(existing, name, allow_missing=True)
    comparable = existing_num.notna() & derived.notna()
    mismatch = comparable & (existing_num.astype("Float64") != derived.astype("Float64"))
    if mismatch.any():
        indices = list(existing.index[mismatch][:10])
        raise EndpointDefinitionError(
            f"{name} disagrees with the manuscript rule in {int(mismatch.sum())} rows; "
            f"first indices: {indices}."
        )


def compute_pcoma_c_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    source = [
        "pcoma_c_exposure_component",
        "pcoma_c_reconstruction_component",
        "temporary_clip_cumulative_sec",
        "clip_reposition_count",
        "flow_rescue_manoeuvre_flag",
        "rupture_escalation_flag",
    ]
    raw_available = all(column in df.columns for column in source)
    existing_available = "derived_high_complexity_case" in df.columns
    if not raw_available and existing_available:
        endpoint = _binary(df["derived_high_complexity_case"], "derived_high_complexity_case")
        return pd.DataFrame({"derived_high_complexity_case": endpoint}, index=df.index)
    _require(df, source, "PComA-C")

    c1 = _binary(df[source[0]], source[0]).astype(int)
    c2 = _binary(df[source[1]], source[1]).astype(int)
    duration = pd.to_numeric(df[source[2]], errors="coerce")
    reposition = pd.to_numeric(df[source[3]], errors="coerce")
    if duration.isna().any() or reposition.isna().any():
        raise EndpointDefinitionError("PComA-C duration and reposition counts must be observed.")
    if (duration < 0).any() or (reposition < 0).any():
        raise EndpointDefinitionError("PComA-C duration and reposition counts cannot be negative.")
    c3 = (duration > 600).astype(int)
    c4 = (reposition >= 3).astype(int)
    c5 = _binary(df[source[4]], source[4]).astype(int)
    c6 = _binary(df[source[5]], source[5]).astype(int)

    component_count = c1 + c2 + c3 + c4 + c5 + c6
    derived = ((component_count >= 2) | (c4 == 1) | (c5 == 1)).astype("Int64")
    if existing_available:
        _validate_existing(df["derived_high_complexity_case"], derived, "derived_high_complexity_case")

    return pd.DataFrame(
        {
            "pcoma_c_component_count": component_count.astype("Int64"),
            "derived_high_complexity_case": derived,
        },
        index=df.index,
    )


def compute_pcoma_r_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    source = [
        "new_postop_dwi_lesion_flag",
        "new_postop_ct_hypodensity_flag",
        "new_postop_ischaemic_deficit_flag",
        "lesion_anatomically_concordant_flag",
        "preexisting_infarct_flag",
        "delayed_cerebral_ischaemia_flag",
        "vasospasm_related_infarct_flag",
        "venous_infarction_flag",
        "nonprocedural_embolic_lesion_flag",
        "ischaemia_mechanism_indeterminate_flag",
    ]
    raw_available = all(column in df.columns for column in source)
    existing_available = "derived_procedure_related_ischemia_flag" in df.columns
    if not raw_available and existing_available:
        endpoint = _binary(
            df["derived_procedure_related_ischemia_flag"],
            "derived_procedure_related_ischemia_flag",
        )
        return pd.DataFrame(
            {"derived_procedure_related_ischemia_flag": endpoint}, index=df.index
        )
    _require(df, source, "PComA-R")

    flags = {column: _binary(df[column], column, allow_missing=True) for column in source}
    imaging = (
        (flags["new_postop_dwi_lesion_flag"] == 1)
        | (flags["new_postop_ct_hypodensity_flag"] == 1)
    ) & (flags["lesion_anatomically_concordant_flag"] == 1)
    clinical = flags["new_postop_ischaemic_deficit_flag"] == 1
    exclusions = (
        (flags["preexisting_infarct_flag"] == 1)
        | (flags["delayed_cerebral_ischaemia_flag"] == 1)
        | (flags["vasospasm_related_infarct_flag"] == 1)
        | (flags["venous_infarction_flag"] == 1)
        | (flags["nonprocedural_embolic_lesion_flag"] == 1)
    )
    indeterminate = flags["ischaemia_mechanism_indeterminate_flag"] == 1

    derived = pd.Series(pd.NA, index=df.index, dtype="Int64")
    sufficient = pd.concat(list(flags.values()), axis=1).notna().all(axis=1)
    derived.loc[sufficient & ~indeterminate] = (
        (imaging | clinical) & ~exclusions
    ).loc[sufficient & ~indeterminate].astype(int)

    if existing_available:
        _validate_existing(
            df["derived_procedure_related_ischemia_flag"],
            derived,
            "derived_procedure_related_ischemia_flag",
        )
        existing = _binary(
            df["derived_procedure_related_ischemia_flag"],
            "derived_procedure_related_ischemia_flag",
            allow_missing=True,
        )
        derived = derived.fillna(existing)

    if derived.isna().any():
        raise EndpointDefinitionError(
            "PComA-R remains indeterminate in one or more rows. Resolve adjudication rather than imputing the endpoint."
        )
    return pd.DataFrame(
        {"derived_procedure_related_ischemia_flag": derived}, index=df.index
    )


def compute_pcoma_o_survival_fields(df: pd.DataFrame) -> pd.DataFrame:
    if "preop_cn3_palsy" not in df.columns:
        raise EndpointDefinitionError("PComA-O requires preop_cn3_palsy.")
    eligible = _binary(df["preop_cn3_palsy"], "preop_cn3_palsy") == 1

    if "cn3_complete_recovery_12m" in df.columns:
        event = _binary(
            df["cn3_complete_recovery_12m"],
            "cn3_complete_recovery_12m",
            allow_missing=True,
        )
    elif "cn3_complete_recovery_event" in df.columns:
        event = _binary(
            df["cn3_complete_recovery_event"],
            "cn3_complete_recovery_event",
            allow_missing=True,
        )
    else:
        raise EndpointDefinitionError(
            "PComA-O requires cn3_complete_recovery_event or cn3_complete_recovery_12m."
        )

    _require(df, ["time_to_complete_recovery_days", "cn3_last_assessment_days"], "PComA-O")
    recovery_time = pd.to_numeric(df["time_to_complete_recovery_days"], errors="coerce")
    last_assessment = pd.to_numeric(df["cn3_last_assessment_days"], errors="coerce")
    death_flag = (
        _binary(df["death_before_cn3_recovery"], "death_before_cn3_recovery", allow_missing=True)
        if "death_before_cn3_recovery" in df.columns
        else pd.Series(0, index=df.index, dtype="Int64")
    )
    death_time = (
        pd.to_numeric(df["death_time_days"], errors="coerce")
        if "death_time_days" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    out_event = pd.Series(pd.NA, index=df.index, dtype="Int64")
    out_time = pd.Series(np.nan, index=df.index, dtype=float)

    for index in df.index[eligible]:
        observed_event = event.loc[index]
        if pd.isna(observed_event):
            raise EndpointDefinitionError(f"PComA-O event indicator is missing at row {index}.")
        if int(observed_event) == 1:
            time_value = recovery_time.loc[index]
            if pd.isna(time_value) or time_value <= 0 or time_value > 365:
                raise EndpointDefinitionError(
                    f"Recovered PComA-O row {index} requires a recovery time in (0, 365]."
                )
            out_event.loc[index] = 1
            out_time.loc[index] = float(time_value)
        else:
            candidates = [365.0]
            if pd.notna(last_assessment.loc[index]) and last_assessment.loc[index] > 0:
                candidates.append(float(last_assessment.loc[index]))
            if death_flag.loc[index] == 1 and pd.notna(death_time.loc[index]) and death_time.loc[index] > 0:
                candidates.append(float(death_time.loc[index]))
            censor_time = min(candidates)
            if censor_time <= 0:
                raise EndpointDefinitionError(f"Invalid PComA-O censor time at row {index}.")
            out_event.loc[index] = 0
            out_time.loc[index] = censor_time

    nonrecovery_12m = pd.Series(pd.NA, index=df.index, dtype="Int64")
    nonrecovery_12m.loc[eligible & event.notna()] = (1 - event.loc[eligible & event.notna()].astype(int)).astype("Int64")
    return pd.DataFrame(
        {
            "pcoma_o_event": out_event,
            "pcoma_o_time_days": out_time,
            "derived_cn3_nonrecovery_12m": nonrecovery_12m,
        },
        index=df.index,
    )


def compute_pcoma_e_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, ["mrs_90d"], "PComA-E")
    mrs = pd.to_numeric(df["mrs_90d"], errors="coerce")
    invalid = mrs.dropna().loc[~mrs.dropna().between(0, 6)]
    if not invalid.empty:
        raise EndpointDefinitionError("mrs_90d contains values outside 0-6.")

    if "mrs_90d_window_eligible_flag" in df.columns:
        eligible = _binary(
            df["mrs_90d_window_eligible_flag"],
            "mrs_90d_window_eligible_flag",
        ) == 1
    elif "mrs_90d_assessment_day" in df.columns:
        day = pd.to_numeric(df["mrs_90d_assessment_day"], errors="coerce")
        eligible = day.between(76, 104)
    else:
        eligible = mrs.notna()

    poor = pd.Series(pd.NA, index=df.index, dtype="Int64")
    poor.loc[eligible & mrs.notna()] = (mrs.loc[eligible & mrs.notna()] > 2).astype(int)
    if "derived_poor_outcome_90d" in df.columns:
        _validate_existing(df["derived_poor_outcome_90d"], poor, "derived_poor_outcome_90d")
        existing = _binary(
            df["derived_poor_outcome_90d"],
            "derived_poor_outcome_90d",
            allow_missing=True,
        )
        poor = poor.fillna(existing.where(eligible))

    return pd.DataFrame(
        {
            "mrs_90d_window_eligible_flag": eligible.astype("Int64"),
            "derived_poor_outcome_90d": poor,
        },
        index=df.index,
    )
