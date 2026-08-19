from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from .config import ExperimentConfig, HyperParameters, get_paper_preset
from .data import (
    cache_identity,
    deterministic_stratified_subset,
    load_prepared,
    prepare_cicids2017,
    prepare_nsl_kdd,
    subset_manifest,
)
from .provenance import load_parameter_selection, preset_provenance
from .reporting import audit_saved_run, write_paper_tables
from .selection import cost_equation, validate_fitness_configuration
from .statistics import mcnemar_result_directories
from .swarm import (
    CachedObjective,
    particle_swarm_search,
    random_search,
    salp_swarm_search,
)
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
    train.add_argument(
        "--selection-source",
        choices=(
            "paper_preset",
            "pso_search",
            "ssa_search",
            "random_search",
            "manual",
            "transferred_cic_preset",
        ),
        required=True,
    )
    train.add_argument("--paper-optimizer", choices=("pso", "ssa"))
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--params-json", type=Path)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=_positive)
    train.add_argument(
        "--protocol",
        choices=("paper_replication", "rigorous_evaluation"),
        default=None,
    )
    train.add_argument("--mode", choices=("smoke", "full"), default="full")
    train.add_argument(
        "--modeling-mode",
        choices=("feature_axis_replication", "temporal_window"),
        default="feature_axis_replication",
    )
    train.add_argument("--threshold", type=float, default=0.5)
    train.add_argument("--optimize-threshold", action="store_true")
    train.add_argument(
        "--selection-fitness",
        choices=("accuracy", "macro_f1", "cost_sensitive"),
    )
    train.add_argument("--selection-patience", type=int, default=10)
    train.add_argument("--selection-min-delta", type=float, default=0.0)
    train.add_argument("--false-positive-cost", type=float, default=1.0)
    train.add_argument("--false-negative-cost", type=float, default=1.0)
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
    search.add_argument("--selection-patience", type=int, default=10)
    search.add_argument("--selection-min-delta", type=float, default=0.0)
    search.add_argument("--checkpoint", type=Path)
    search.add_argument("--resume", action="store_true")
    search.add_argument("--pso-velocity-clip", type=float)
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

    mcnemar = commands.add_parser(
        "mcnemar", help="Compare two runs using aligned prediction-level McNemar testing"
    )
    mcnemar.add_argument("--first-run", type=Path, required=True)
    mcnemar.add_argument("--second-run", type=Path, required=True)
    mcnemar.add_argument("--output", type=Path, required=True)

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
        "cache_dir": str(args.cache_dir.resolve()),
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
    source = args.selection_source
    selection_fitness = args.selection_fitness or (
        "accuracy" if source == "paper_preset" else "macro_f1"
    )
    if source in {"paper_preset", "transferred_cic_preset"}:
        if args.params_json is not None:
            raise ValueError(f"{source} cannot be combined with --params-json")
        if args.paper_optimizer is None:
            raise ValueError(f"{source} requires --paper-optimizer pso|ssa")
        params = get_paper_preset(args.task, args.model, args.paper_optimizer)
        provenance = preset_provenance(
            selection_source=source,
            paper_optimizer=args.paper_optimizer,
            dataset=dataset,
            task=args.task,
            model=args.model,
            seed=args.seed,
            params=params,
        )
    else:
        if args.params_json is None:
            raise ValueError(f"selection_source={source} requires --params-json")
        if args.paper_optimizer is not None:
            raise ValueError("--paper-optimizer is only valid for a paper preset source")
        params, provenance = load_parameter_selection(
            args.params_json,
            selection_source=source,
            dataset=dataset,
            task=args.task,
            model=args.model,
            seed=args.seed,
            expected_fitness=selection_fitness,
        )
    protocol = args.protocol or (
        "paper_replication" if source == "paper_preset" else "rigorous_evaluation"
    )
    validate_fitness_configuration(
        args.task,
        selection_fitness,
        args.false_positive_cost,
        args.false_negative_cost,
    )
    if args.selection_patience < 0:
        raise ValueError("selection_patience cannot be negative")
    if not math.isfinite(args.selection_min_delta) or args.selection_min_delta < 0:
        raise ValueError("selection_min_delta must be finite and non-negative")
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
        params=params,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs_override=epochs,
        max_train_samples=max_train,
        max_val_samples=max_val,
        max_test_samples=max_test,
        verbose=args.verbose,
        protocol=protocol,
        modeling_mode=args.modeling_mode,
        threshold=args.threshold,
        optimize_threshold=args.optimize_threshold,
        run_mode=args.mode,
        selection_provenance=provenance,
        selection_fitness=selection_fitness,
        selection_patience=args.selection_patience,
        selection_min_delta=args.selection_min_delta,
        false_positive_cost=args.false_positive_cost,
        false_negative_cost=args.false_negative_cost,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _search(args: argparse.Namespace) -> int:
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite search result: {args.output}")
    if args.resume and args.output.exists():
        raise FileExistsError(
            f"Completed result already exists: {args.output}; resume uses the checkpoint"
        )
    dataset = load_prepared(args.cache_dir)
    validate_fitness_configuration(
        args.task,
        args.fitness,
        args.false_positive_cost,
        args.false_negative_cost,
    )
    if args.selection_patience < 0:
        raise ValueError("selection_patience cannot be negative")
    if not math.isfinite(args.selection_min_delta) or args.selection_min_delta < 0:
        raise ValueError("selection_min_delta must be finite and non-negative")
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

    train_labels = dataset.labels("train", args.task)
    validation_labels = dataset.labels("val", args.task)
    train_indices = deterministic_stratified_subset(train_labels, max_train, args.seed)
    validation_indices = deterministic_stratified_subset(
        validation_labels, max_val, args.seed + 1
    )
    identity = cache_identity(dataset)
    search_contract = {
        "dataset": dataset.metadata["dataset"],
        "cache_dir": str(args.cache_dir.resolve()),
        "cache_identity": identity,
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
        "training_subset": subset_manifest(train_indices, train_labels),
        "validation_subset": subset_manifest(validation_indices, validation_labels),
        "test_set_accessed": False,
        "seed": args.seed,
        "candidate_seed_policy": "fixed seed for every candidate",
        "modeling_mode": args.modeling_mode,
        "false_positive_cost": args.false_positive_cost,
        "false_negative_cost": args.false_negative_cost,
        "cost_equation": (
            cost_equation(args.false_positive_cost, args.false_negative_cost)
            if args.fitness == "cost_sensitive"
            else None
        ),
        "selection_patience": args.selection_patience,
        "selection_min_delta": args.selection_min_delta,
        "pso_velocity_clip": args.pso_velocity_clip,
    }

    def uncached_objective(params: HyperParameters, evaluation: int) -> dict:
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
            patience=args.selection_patience,
            min_delta=args.selection_min_delta,
            return_details=True,
        )

    objective = CachedObjective(
        uncached_objective,
        key_context={
            "dataset_split_checksum": identity.get("split_checksum"),
            "cache_identity_sha256": identity["identity_sha256"],
            "task": args.task,
            "model": args.model,
            "training_subset_checksum": search_contract["training_subset"][
                "indices_checksum_sha256"
            ],
            "validation_subset_checksum": search_contract["validation_subset"][
                "indices_checksum_sha256"
            ],
            "seed": args.seed,
            "epoch_cap": epoch_cap,
            "fitness": args.fitness,
            "false_positive_cost": args.false_positive_cost,
            "false_negative_cost": args.false_negative_cost,
        },
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
    checkpoint = args.checkpoint or args.output.with_suffix(".checkpoint.json")
    optimizer_options = {}
    if args.algorithm == "pso":
        optimizer_options["velocity_clip"] = args.pso_velocity_clip
    result = functions[args.algorithm](
        objective,
        model_name=args.model,
        population_size=population_size,
        iterations=iterations,
        seed=args.seed,
        progress=progress,
        checkpoint_path=checkpoint,
        resume=args.resume,
        run_identity=search_contract,
        **optimizer_options,
    )
    result.metadata.update(search_contract)
    result.metadata["checkpoint_path"] = str(checkpoint.resolve())
    result.metadata.update(
        {
            key: value
            for key, value in result.to_dict().items()
            if key
            in {
                "proposal_count",
                "unique_configuration_count",
                "cache_hits",
                "nn_fit_attempts",
                "completed_nn_fits",
                "failed_candidates",
            }
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
                selection_patience=config.selection_patience,
                selection_min_delta=config.selection_min_delta,
                checkpoint=Path(config.output_dir) / "search_checkpoint.json",
                resume=config.resume,
                pso_velocity_clip=config.pso_velocity_clip,
            )
        )
    dataset = load_prepared(config.cache_dir)
    if config.selection_source in {"paper_preset", "transferred_cic_preset"}:
        params = get_paper_preset(config.task, config.model, config.swarm_preset)
        provenance = preset_provenance(
            selection_source=config.selection_source,
            paper_optimizer=config.swarm_preset,
            dataset=dataset,
            task=config.task,
            model=config.model,
            seed=config.seed,
            params=params,
        )
    elif config.parameters_json:
        params, provenance = load_parameter_selection(
            config.parameters_json,
            selection_source=config.selection_source,
            dataset=dataset,
            task=config.task,
            model=config.model,
            seed=config.seed,
            expected_fitness=config.fitness,
        )
    elif config.hyperparameters:
        params = HyperParameters(**config.hyperparameters)
        provenance = {
            "selection_source": config.selection_source,
            "algorithm": None,
            "seed": config.seed,
            "dataset": dataset.metadata["dataset"],
            "task": config.task,
            "model": config.model,
            "cache_identity": cache_identity(dataset),
            "decoded_parameters": params.to_dict(),
        }
    else:
        raise ValueError("Fixed runs require an explicit parameter selection source")
    result = train_and_evaluate(
        dataset,
        task=config.task,
        model_name=config.model,
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
        selection_provenance=provenance,
        selection_fitness=config.fitness,
        selection_patience=config.selection_patience,
        selection_min_delta=config.selection_min_delta,
        false_positive_cost=config.false_positive_cost,
        false_negative_cost=config.false_negative_cost,
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
    if args.command == "mcnemar":
        result = mcnemar_result_directories(
            args.first_run, args.second_run, args.output
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run-config":
        return _run_config(args)
    raise AssertionError(args.command)
