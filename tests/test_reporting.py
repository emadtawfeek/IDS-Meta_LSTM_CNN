import numpy as np

from ids_repro.reporting import paper_matrix_metrics


def test_paper_multiclass_metrics_do_not_mix_averaging_schemes():
    metrics = paper_matrix_metrics()["multiclass"]["ssa"]["metrics"]
    assert np.isclose(metrics["accuracy"], 0.9973659155823548)
    assert np.isclose(metrics["precision_macro"], 0.9702806338138605)
    assert np.isclose(metrics["recall_macro"], 0.8735203431341806)
    assert np.isclose(metrics["f1_macro"], 0.9073788514198186)
    assert not np.isclose(metrics["f1_macro"], metrics["f1_micro"])
