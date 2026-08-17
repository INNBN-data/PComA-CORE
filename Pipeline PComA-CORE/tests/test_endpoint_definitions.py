import pandas as pd
import pytest

from preprocessing.endpoint_definitions import (
    EndpointDefinitionError,
    compute_pcoma_c_endpoint,
    compute_pcoma_e_endpoints,
)


def test_pcoma_c_rule():
    df = pd.DataFrame({
        "pcoma_c_exposure_component": [1, 0, 0],
        "pcoma_c_reconstruction_component": [1, 0, 0],
        "temporary_clip_cumulative_sec": [0, 0, 0],
        "clip_reposition_count": [0, 3, 0],
        "flow_rescue_manoeuvre_flag": [0, 0, 1],
        "rupture_escalation_flag": [0, 0, 0],
    })
    out = compute_pcoma_c_endpoint(df)
    assert out["derived_high_complexity_case"].tolist() == [1, 1, 1]


def test_pcoma_e_missing_outcome_stays_missing():
    df = pd.DataFrame({"mrs_90d": [0, 3, None]})
    out = compute_pcoma_e_endpoints(df)
    assert out["derived_poor_outcome_90d"].iloc[0] == 0
    assert out["derived_poor_outcome_90d"].iloc[1] == 1
    assert pd.isna(out["derived_poor_outcome_90d"].iloc[2])


def test_pcoma_c_missing_sources_fails_without_endpoint():
    with pytest.raises(EndpointDefinitionError):
        compute_pcoma_c_endpoint(pd.DataFrame({"temporary_clip_cumulative_sec": [0]}))
