# Corrected-project change manifest

No file under `../author-original`, no raw dataset, no paper PDF, no existing cache,
and no pre-audit artifact under `outputs/` was modified.

## Modified corrected-project files

- `pyproject.toml`
- `README.md`
- `src/ids_repro/config.py`
- `src/ids_repro/data.py`
- `src/ids_repro/models.py`
- `src/ids_repro/swarm.py`
- `src/ids_repro/metrics.py`
- `src/ids_repro/training.py`
- `src/ids_repro/cli.py`
- `tests/test_config.py`
- `tests/test_metrics.py`
- `tests/test_swarm.py`

## New implementation and documentation files

- `requirements.txt`
- `AUDIT_AND_REPRODUCTION_REPORT.md`
- `LEGACY_IMPLEMENTATIONS.md`
- `CHANGE_MANIFEST.md`
- `src/ids_repro/protocols.py`
- `src/ids_repro/reporting.py`
- `src/ids_repro/statistics.py`
- `src/ids_repro/datasets/__init__.py`
- `src/ids_repro/datasets/common.py`
- `src/ids_repro/datasets/cicids2017.py`
- `src/ids_repro/datasets/nsl_kdd.py`
- `src/ids_repro/modeling/__init__.py`
- `src/ids_repro/modeling/cnn.py`
- `src/ids_repro/modeling/lstm.py`
- `src/ids_repro/modeling/cnn_lstm.py`
- `src/ids_repro/optimization/__init__.py`
- `src/ids_repro/optimization/search_space.py`
- `src/ids_repro/optimization/pso.py`
- `src/ids_repro/optimization/ssa.py`
- `src/ids_repro/optimization/random_search.py`
- `src/ids_repro/evaluation/__init__.py`
- `src/ids_repro/evaluation/metrics.py`
- `src/ids_repro/evaluation/reporting.py`
- `src/ids_repro/evaluation/statistical_tests.py`
- `src/ids_repro/engine/__init__.py`
- `src/ids_repro/engine/trainer.py`
- `src/ids_repro/engine/fitness.py`
- `src/ids_repro/engine/reproducibility.py`
- `configs/cicids2017-paper-binary-ssa.yaml`
- `configs/cicids2017-rigorous-binary-ssa.yaml`
- `configs/nsl-kdd-rigorous-multiclass-ssa.yaml`
- `scripts/generate_inventory.py`
- `scripts/summarize_repeated_runs.py`
- `tests/test_protocols.py`
- `tests/test_data_shapes.py`
- `tests/test_models.py`
- `tests/test_preprocessing.py`
- `tests/test_reporting.py`
- `tests/test_statistics.py`
- `tests/test_training_guards.py`

## New generated evidence

- `cache/nsl-kdd-rigorous/` — separately prepared, stratified NSL-KDD extension cache
- `results/audit/FILE_INVENTORY.json`
- `results/audit/submitted-cic-binary-ssa/`
- `results/audit/nsl-kdd-rigorous-smoke/`
- `results/audit/nsl-kdd-rigorous-smoke-final/`
- `results/paper-recalculation/` and `results/paper-recalculation-final/`
- `results/cicids2017/binary/cnn-lstm/fixed_ssa/seed_42_smoke/`
- `results/cicids2017/binary/cnn/{pso,ssa,random}/seed_42_smoke/`
- `results/cicids2017/binary/cnn/pso/seed_42_smoke_final/`
- `results/nsl-kdd/multiclass/cnn-lstm/fixed_ssa/seed_42_smoke/`
- `results/smoke/` — earlier bounded development artifacts retained for provenance

Generated results are never hand-edited. Numerical comparison tables are produced
by `ids_repro.reporting` from saved arrays and confusion matrices.
