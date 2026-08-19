from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import ExperimentConfig, HyperParameters, get_paper_preset
from .data import load_prepared, prepare_cicids2017, prepare_nsl_kdd
from .reporting import audit_saved_run, write_paper_tables
from .swarm import particle_swarm_search, random_search, salp_swarm_search
from .training import train_and_evaluate, validation_fitness


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _optional_positive(value: str) -> int | None:
    if value.lower() == "none":
        return None
    return _positive(value)


def _add_sample_limits(parser: argparse.ArgumentParser, include_test: bool = True) -> None:
    parser.add_argument("--max-train-samples", type=_optional_positive)
    parser.add_argument("--max-val-samples", type=_optional_positive)
    if include_test:
        parser.add_argument("--max-test-samples", type=_optional_positive)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ids-repro",
        description="Audited PSO/SSA CNN-LSTM intrusion-detection reproduction.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Clean, split, scale, and cache a dataset")
    prepare.add_argument("--dataset", choices=("cicids2017", "nsl-kdd"), required=True)
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--cache-dir", type=Path, required=True)
    prepare.add_argument("--chunk-size", type=_positive, default=100_000)
    prepare.add_argument(
        "--protocol",
        choices=("paper_replication", "rigorous_evaluation"),
        default=None,
    )
    prepare.add_argument("--force", action="store_true")

    verify = commands.add_parser("verify", help="Verify cached shapes and preprocessing")
    verify.add_argument("--cache-dir", type=Path, required=True)

    train = commands.add_parser("train", help="Train fixed parameters, then test exactly once")
    train.add_argument("--cache-dir", type=Path, required=True)
    train.add_argument("--task", choices=("binary", "multiclass"), required=True)
    train.add_argument("--model", choices=("cnn", "lstm", "cnn-lstm"), required=True)
    train.add_argument("--swarm", choices=("pso", "ssa"), required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--params-json", type=Path)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=_positive)
    train.add_argument(
        "--protocol",
        choices=("paper_replication", "rigorous_evaluation"),
        default="paper_replication",
    )
    train.add_argument("--mode", choices=("smoke", "full"), default="full")
    train.add_argument(
        "--modeling-mode",
        choices=("feature_axis_replication", "temporal_window"),
        default="feature_axis_replication",
    )
    train.add_argument("--threshold", type=float, default=0.5)
    train.add_argument("--optimize-threshold", action="store_true")
    train.add_argument("--verbose", type=int, choices=(0, 1, 2), default=2)
    _add_sample_limits(train)

    search = commands.add_parser(
        "search", help="Run corrected PSO, SSA, or budget-matched random search"
    )
    search.add_argument("--cache-dir", type=Path, required=True)
    search.add_argument("--task", choices=("binary", "multiclass"), required=True)
    search.add_argument("--model", choices=("cnn", "lstm", "cnn-lstm"), required=True)
    search.add_argument("--algorithm", choices=("pso", "ssa", "random"), required=True)
    search.add_argument("--output", type=Path, required=True)
    search.add_argument("--mode", choices=("smoke", "full"), default="full")
    search.add_argument("--population-size", type=_positive)
    search.add_argument("--iterations", type=_positive)
    search.add_argument("--seed", type=int, default=42)
    search.add_argument("--epoch-cap", type=_optional_positive)
    search.add_argument(
        "--fitness",
        choices=("accuracy", "macro_f1", "cost_sensitive"),
        default="macro_f1",
    )
    search.add_argument("--false-positive-cost", type=float, default=1.0)
    search.add_argument("--false-negative-cost", type=float, default=1.0)
    search.add_argument(
        "--modeling-mode",
        choices=("feature_axis_replication", "temporal_window"),
        default="feature_axis_replication",
    )
    _add_sample_limits(search, include_test=False)

    audit = commands.add_parser(
        "audit-artifact", help="Recalculate a saved run without changing it"
    )
    audit.add_argument("--run-dir", type=Path, required=True)
    audit.add_argument("--cache-dir", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--task", choices=("binary", "multiclass"), required=True)
    audit.add_argument("--paper-algorithm", choices=("pso", "ssa"), required=True)

    tables = commands.add_parser(
        "paper-report", help="Recalculate aggregate metrics from the published matrices"
    )
    tables.add_argument("--output-dir", type=Path, required=True)
    tables.add_argument("--submitted-audit", type=Path)

    configured = commands.add_parser("run-config", help="Run a fixed or search experiment from YAML")
    configured.add_argument("--config", type=Path, required=True)
    return parser


def _load_params(path: Path) -> HyperParameters:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "best_parameters" in payload:
        payload = payload["best_parameters"]
    return HyperParameters(**payload)


def _prepare(args: argparse.Namespace) -> int:
    protocol = args.protocol or (
        "paper_replication"
        if args.dataset == "cicids2017"
        else "rigorous_evaluation"
    )
    if args.dataset == "cicids2017":
        dataset = prepare_cicids2017(
            args.data_dir,
            args.cache_dir,
            force=args.force,
            chunk_size=args.chunk_size,
            protocol=protocol,
        )
    else:
        dataset = prepare_nsl_kdd(
            args.data_dir, args.cache_dir, force=args.force, protocol=protocol
        )
    print(json.dumps(dataset.metadata, indent=2, sort_keys=True))
    return 0


def _verify(args: argparse.Namespace) -> int:
    import numpy as np

    dataset = load_prepared(args.cache_dir)
    finite = all(
        np.isfinite(part).all()
        for part in (dataset.train_x, dataset.val_x, dataset.test_x)
    )
    checks = {
        "dataset": dataset.metadata["dataset"],
        "protocol": dataset.metadata.get("protocol", "legacy cache; see metadata"),
        "train_shape": list(dataset.train_x.shape),
        "validation_shape": list(dataset.val_x.shape),
        "test_shape": list(dataset.test_x.shape),
        "finite": bool(finite),
        "feature_count_consistent": bool(
            dataset.train_x.shape[-1]
            == dataset.val_x.shape[-1]
            == dataset.test_x.shape[-1]
            == len(dataset.feature_names)
        ),
        "range": {
            "train_min": float(dataset.train_x.min()),
            "train_max": float(dataset.train_x.max()),
            "validation_min": float(dataset.val_x.min()),
            "validation_max": float(dataset.val_x.max()),
            "test_min": float(dataset.test_x.min()),
            "test_max": float(dataset.test_x.max()),
        },
    }
    if dataset.metadata["dataset"] == "cicids2017":
        checks["paper_split_match"] = bool(dataset.metadata.get("paper_split_match"))
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if finite and checks["feature_count_consistent"] else 2


def _train(args: argparse.Namespace) -> int:
    dataset = load_prepared(args.cache_dir)
    params = (
        _load_params(args.params_json)
        if args.params_json
        else get_paper_preset(args.task, args.model, args.swarm)
    )
    epochs = args.epochs if args.epochs is not None else (1 if args.mode == "smoke" else None)
    max_train = (
        args.max_train_samples
        if args.max_train_samples is not None
        else (20_000 if args.mode == "smoke" else None)
    )
    max_val = (
        args.max_val_samples
        if args.max_val_samples is not None
        else (5_000 if args.mode == "smoke" else None)
    )
    max_test = (
        args.max_test_samples
        if args.max_test_samples is not None
        else (5_000 if args.mode == "smoke" else None)
    )
    result = train_and_evaluate(
        dataset,
        task=args.task,
        model_name=args.model,
        swarm_name=args.swarm,
        params=params,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs_override=epochs,
        max_train_samples=max_train,
        max_val_samples=max_val,
        max_test_samples=max_test,
        verbose=args.verbose,
        protocol=args.protocol,
        modeling_mode=args.modeling_mode,
        threshold=args.threshold,
        optimize_threshold=args.optimize_threshold,
        run_mode=args.mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _search(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite search result: {args.output}")
    dataset = load_prepared(args.cache_dir)
    population_size = args.population_size or (3 if args.mode == "smoke" else 10)
    iterations = args.iterations or (2 if args.mode == "smoke" else 100)
    epoch_cap = args.epoch_cap if args.epoch_cap is not None else (2 if args.mode == "smoke" else None)
    max_train = (
        args.max_train_samples
        if args.max_train_samples is not None
        else (20_000 if args.mode == "smoke" else None)
    )
    max_val = (
        args.max_val_samples
        if args.max_val_samples is not None
        else (5_000 if args.mode == "smoke" else None)
    )

    def objective(params: HyperParameters, evaluation: int) -> float:
        return validation_fitness(
            dataset,
            task=args.task,
            model_name=args.model,
            params=params,
            seed=args.seed,
            epoch_cap=epoch_cap,
            max_train_samples=max_train,
            max_val_samples=max_val,
            fitness_name=args.fitness,
            false_positive_cost=args.false_positive_cost,
            false_negative_cost=args.false_negative_cost,
            modeling_mode=args.modeling_mode,
        )

    def progress(iteration: int, fitness: float) -> None:
        print(
            f"iteration={iteration}/{iterations} "
            f"best_validation_{args.fitness}={fitness:.8f}"
        )

    functions = {
        "pso": particle_swarm_search,
        "ssa": salp_swarm_search,
        "random": random_search,
    }
    result = functions[args.algorithm](
        objective,
        model_name=args.model,
        population_size=population_size,
        iterations=iterations,
        seed=args.seed,
        progress=progress,
    )
    result.metadata.update(
        {
            "dataset": dataset.metadata["dataset"],
            "cache_dir": str(args.cache_dir.resolve()),
            "task": args.task,
            "model": args.model,
            "algorithm": args.algorithm,
            "run_mode": args.mode,
            "fitness": args.fitness,
            "population_size": population_size,
            "iterations": iterations,
            "evaluation_budget": population_size * iterations,
            "epoch_cap": epoch_cap,
            "max_train_samples": max_train,
            "max_validation_samples": max_val,
            "test_set_accessed": False,
            "seed": args.seed,
            "modeling_mode": args.modeling_mode,
            "false_positive_cost": args.false_positive_cost,
            "false_negative_cost": args.false_negative_cost,
        }
    )
    for record in result.evaluation_trace:
        record["candidate_seed"] = args.seed
    result.save(args.output)
    artifact_dir = args.output.with_suffix("")
    result.save_artifacts(artifact_dir)
    cache_metadata = args.cache_dir / "metadata.json"
    if cache_metadata.exists():
        shutil.copy2(cache_metadata, artifact_dir / "dataset_manifest.json")
    preprocessor = args.cache_dir / "preprocessor.joblib"
    if preprocessor.exists():
        shutil.copy2(preprocessor, artifact_dir / "preprocessor.joblib")
    label_encoder = args.cache_dir / "label_encoder.joblib"
    if label_encoder.exists():
        shutil.copy2(label_encoder, artifact_dir / "label_encoder.joblib")
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _run_config(args: argparse.Namespace) -> int:
    config = ExperimentConfig.load(args.config)
    if config.optimizer != "fixed":
        return _search(
            argparse.Namespace(
                cache_dir=Path(config.cache_dir),
                task=config.task,
                model=config.model,
                algorithm=config.optimizer,
                output=Path(config.output_dir) / "search_result.json",
                mode=config.run_mode,
                population_size=config.population_size,
                iterations=config.iterations,
                seed=config.seed,
                epoch_cap=config.epoch_cap,
                fitness=config.fitness,
                false_positive_cost=config.false_positive_cost,
                false_negative_cost=config.false_negative_cost,
                modeling_mode=config.modeling_mode,
                max_train_samples=config.max_train_samples,
                max_val_samples=config.max_val_samples,
            )
        )
    dataset = load_prepared(config.cache_dir)
    params = (
        HyperParameters(**config.hyperparameters)
        if config.hyperparameters
        else get_paper_preset(config.task, config.model, config.swarm_preset)
    )
    result = train_and_evaluate(
        dataset,
        task=config.task,
        model_name=config.model,
        swarm_name=config.swarm_preset,
        params=params,
        output_dir=config.output_dir,
        seed=config.seed,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        max_test_samples=config.max_test_samples,
        protocol=config.protocol,
        modeling_mode=config.modeling_mode,
        threshold=config.threshold,
        deterministic_ops=config.deterministic_ops,
        run_mode=config.run_mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "verify":
        return _verify(args)
    if args.command == "train":
        return _train(args)
    if args.command == "search":
        return _search(args)
    if args.command == "audit-artifact":
        result = audit_saved_run(
            args.run_dir,
            args.cache_dir,
            args.output_dir,
            task=args.task,
            paper_algorithm=args.paper_algorithm,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "paper-report":
        audit = (
            json.loads(args.submitted_audit.read_text(encoding="utf-8"))
            if args.submitted_audit
            else None
        )
        result = write_paper_tables(args.output_dir, audit)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run-config":
        return _run_config(args)
    raise AssertionError(args.command)
