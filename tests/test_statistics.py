import numpy as np

from ids_repro.statistics import paired_comparison, summarize


def test_summary_reports_sample_standard_deviation_and_ci():
    result = summarize([0.8, 0.9, 1.0])
    assert result["n"] == 3
    assert np.isclose(result["mean"], 0.9)
    assert np.isclose(result["standard_deviation"], 0.1)
    assert result["confidence_interval"][0] < 0.9 < result["confidence_interval"][1]


def test_paired_comparison_reports_effect_and_named_test():
    result = paired_comparison(
        [0.91, 0.92, 0.90, 0.93, 0.94],
        [0.89, 0.91, 0.88, 0.90, 0.92],
    )
    assert result["n"] == 5
    assert result["selected_test"] in {"paired_t_test", "wilcoxon_signed_rank"}
    assert "effect_size_cohen_dz" in result
