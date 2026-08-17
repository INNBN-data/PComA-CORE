import yaml


def test_locked_core_counts_and_metrics():
    with open("results/reported_metrics.yaml", "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert data["cohort"]["total"] == 687
    assert data["cohort"]["development"] == 564
    assert data["cohort"]["temporal"] == 123
    assert data["pcoma_c"]["temporal_auc"] == 0.943
    assert data["pcoma_r"]["temporal_auc"] == 0.703
    assert data["pcoma_o"]["selected_landmark_auc"] == 0.902
    assert data["pcoma_e"]["binary_temporal_auc"] == 0.878
    assert data["pcoma_r"]["sparse_score_declared_range"] == [0, 17]
