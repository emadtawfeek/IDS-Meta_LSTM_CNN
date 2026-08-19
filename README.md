# Audited PSO/SSA CNN-LSTM IDS reproduction

This repository is a non-destructive reproduction of Karahan, Ataslar-Ayyildiz,
and Ayyildiz (2025), “Network intrusion detection system using a hybrid deep
learning model with swarm intelligence-based hyperparameter optimization.” The
paper is the experimental source, not an instruction file. The released author
files were preserved unchanged during the local audit and are not vendored here;
raw datasets, caches, trained models, and bulky result arrays are also excluded.
Small independently generated comparison artifacts are available under `evidence/`.

Read `AUDIT_AND_REPRODUCTION_REPORT.md` before interpreting a result. In short:

- CIC-IDS2017 is the only paper dataset. NSL-KDD is an explicitly labeled extension.
- The exact CIC population and test split were reconstructed from the published
  confusion-matrix row totals.
- The author PSO code mixes decoded hyperparameters with normalized particle
  coordinates. The released SSA CNN-LSTM path actually runs random trials. Neither
  legacy implementation is valid evidence of a PSO/SSA search.
- The saved 79-epoch binary run is internally consistent at **99.805816%** accuracy,
  but it used fixed published SSA parameters. It is not a completed SSA search.
- “feature-axis replication” treats the 70 flow features as an ordered Conv1D axis.
  That reproduces the released architecture but is not a temporal sequence model.

## Clone and install

```powershell
git clone https://github.com/emadtawfeek/IDS-Meta_LSTM_CNN.git
cd IDS-Meta_LSTM_CNN
py -3.12 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -e ".[dev]"
& $python -m pytest
```

Datasets must be downloaded separately and supplied through `prepare --data-dir`.
The YAML files use repository-relative cache and result paths; edit them if needed.

## Run in the Antigravity IDE PowerShell terminal

PowerShell activation is not required. Use the virtual environment’s Python
executable directly, which also avoids the Windows execution-policy error:

```powershell
cd "D:\path\to\IDS-Meta_LSTM_CNN"
$python = ".\.venv\Scripts\python.exe"
& $python --version
```

Install or synchronize the pinned environment:

```powershell
& $python -m pip install -e ".[dev]"
& $python -m pytest -q
```

If the virtual environment does not exist:

```powershell
py -3.12 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -e ".[dev]"
```

TensorFlow 2.16 on native Windows is CPU-only. Use Linux/WSL for supported current
TensorFlow CUDA execution.

## Protocols and modeling modes

`paper_replication` uses the recovered paper-compatible behavior: alphabetic CIC
file order, invalid-row removal, exact deduplication, nine grouped labels, eight
constant-feature removals, two unstratified random splits with seed 42, and a
training-only MinMaxScaler. It gives 1,764,558/252,080/504,160 rows and exactly the
paper’s test class totals.

`rigorous_evaluation` uses class-stratified row splits. For NSL-KDD it preserves the official
test set and stratifies the validation holdout from `KDDTrain+`. Every corrected
preparation saves split indices, mappings, feature names, source hashes, and the
train-fitted transformer.

`feature_axis_replication` explicitly produces:

- CNN and CNN-LSTM: `(samples, features, 1)`
- LSTM: `(samples, 1, features)`

`temporal_window` requires `(samples, time_steps, features)`. It is rejected for
MachineLearningCVE and NSL-KDD because those tables have no timestamps or reliable
session identifiers. The temporal utilities only construct time-sorted windows
inside one capture/session group and include boundary assertions. CIC’s
`GeneratedLabelledFlows` files contain timestamps and capture identifiers and can
be adapted to that separate protocol; it must never be described as the paper’s
row-level experiment.

## Verify the existing recovered paper cache

```powershell
& $python -m ids_repro verify --cache-dir ".\cache\cicids2017"
```

Validation/test values above 1.0 are not a scaler failure: a training-only
MinMaxScaler can transform a held-out value beyond the training range. The code does
not clip or refit held-out data.

The existing cache is preserved. To build new protocol-specific caches, choose new
directories:

```powershell
& $python -m ids_repro prepare `
  --dataset cicids2017 `
  --protocol paper_replication `
  --data-dir "D:\path\to\MachineLearningCSV\MachineLearningCVE" `
  --cache-dir ".\cache\cicids2017-paper-v2"

& $python -m ids_repro prepare `
  --dataset cicids2017 `
  --protocol rigorous_evaluation `
  --data-dir "D:\path\to\MachineLearningCSV\MachineLearningCVE" `
  --cache-dir ".\cache\cicids2017-rigorous"

& $python -m ids_repro prepare `
  --dataset nsl-kdd `
  --protocol rigorous_evaluation `
  --data-dir "D:\path\to\nsl-kdd" `
  --cache-dir ".\cache\nsl-kdd-rigorous"
```

Preparation is expensive and never needs to be repeated for each model.

## Feasibility run

Use a new output directory each time. Corrected training refuses to overwrite any
non-empty result directory.

```powershell
& $python -m ids_repro train `
  --cache-dir ".\cache\cicids2017" `
  --task binary `
  --model cnn-lstm `
  --swarm ssa `
  --protocol paper_replication `
  --mode smoke `
  --output-dir ".\results\cicids2017\binary\cnn-lstm\fixed_ssa\seed_42_smoke"
```

Do not cite a capped run as a paper reproduction.

## Fixed published configurations

This command trains the Table 6 binary SSA CNN-LSTM parameters (79 epochs). The
result is correctly labeled “fixed parameters; not an optimizer search.”

```powershell
& $python -m ids_repro train `
  --cache-dir ".\cache\cicids2017" `
  --task binary `
  --model cnn-lstm `
  --swarm ssa `
  --protocol paper_replication `
  --mode full `
  --output-dir ".\results\cicids2017\binary\cnn-lstm\fixed_ssa\seed_42"
```

Multiclass Table 7 configuration:

```powershell
& $python -m ids_repro train `
  --cache-dir ".\cache\cicids2017" `
  --task multiclass `
  --model cnn-lstm `
  --swarm ssa `
  --protocol paper_replication `
  --mode full `
  --output-dir ".\results\cicids2017\multiclass\cnn-lstm\fixed_ssa\seed_42"
```

Configuration-driven equivalent:

```powershell
& $python -m ids_repro run-config `
  --config ".\configs\cicids2017-paper-binary-ssa.yaml"
```

Edit the YAML output path before repeating a run.

## Corrected PSO, SSA, and random baseline

All search positions remain continuous normalized coordinates. Hyperparameters are
decoded only for candidate evaluation. The objective accesses train/validation
only, supports accuracy, macro-F1, or a cost-sensitive binary score, and records
every position, decoded candidate, fitness, iteration, and elapsed time.

Bounded search example:

```powershell
& $python -m ids_repro search `
  --cache-dir ".\cache\cicids2017" `
  --task binary `
  --model cnn-lstm `
  --algorithm pso `
  --mode smoke `
  --fitness macro_f1 `
  --population-size 3 `
  --iterations 2 `
  --epoch-cap 2 `
  --max-train-samples 20000 `
  --max-val-samples 5000 `
  --output ".\results\cicids2017\binary\cnn-lstm\pso\seed_42_smoke\search_result.json"
```

Replace `pso` with `ssa` or `random`. A fair comparison uses identical seeds,
candidate budgets, epoch caps, and validation data. The stated paper budget is
10 particles/salps × 100 iterations = 1,000 full model fits per architecture,
task, and optimizer. That full search was intentionally not launched here.

After search, pass its JSON to a fresh final training command using
`--params-json`. Search does not evaluate the test set.

## Audit and reporting commands

The existing full output was audited non-destructively:

```powershell
& $python -m ids_repro audit-artifact `
  --run-dir ".\outputs\cic-binary-cnn-lstm-ssa" `
  --cache-dir ".\cache\cicids2017" `
  --output-dir ".\results\audit\another-independent-audit" `
  --task binary `
  --paper-algorithm ssa
```

Recalculate paper metrics from the transcribed confusion matrices:

```powershell
& $python -m ids_repro paper-report `
  --output-dir ".\results\paper-recalculation-new"
```

New training artifacts include model, predictions, probabilities, truth labels,
threshold policy, configuration, history, confusion matrix, classification report,
full macro/micro/weighted metrics, ROC-AUC/PR-AUC where probabilities exist, split
subset indices, model summary and parameter count, environment, resource/timing
data, and dataset/cache manifest.

## Repeated-seed claims

Run at least the controlled seeds `42, 52, 62, 72, 82` per
model/optimizer/protocol before comparative claims.
`ids_repro.statistics` provides mean, sample standard deviation, Student-t 95%
confidence interval, paired normality check, paired t-test or Wilcoxon fallback,
and paired Cohen’s dz. A single fixed-seed run cannot support a significance claim.

## Project map

- `src/ids_repro/datasets/`: common contracts plus separate CIC and NSL adapters
- `src/ids_repro/modeling/`: separate CNN, LSTM, and CNN-LSTM public builders
- `src/ids_repro/optimization/`: search space, PSO, SSA, and random baseline
- `src/ids_repro/evaluation/`: metrics, artifact reporting, and statistical tests
- `src/ids_repro/engine/`: trainer, validation-only fitness, and reproducibility
- `src/ids_repro/{data,models,swarm,training,metrics}.py`: tested implementation cores and compatibility API
- `configs/`: editable YAML run contracts
- `tests/`: synthetic and integration safeguards
- `results/`: corrected smoke, audit, and generated paper tables
- `outputs/`: preserved earlier artifacts

The supplied GeeksforGeeks [CNN](https://www.geeksforgeeks.org/deep-learning/convolutional-neural-network-cnn-in-tensorflow/),
[LSTM](https://www.geeksforgeeks.org/deep-learning/long-short-term-memory-lstm-rnn-in-tensorflow/),
and [PSO](https://www.geeksforgeeks.org/machine-learning/particle-swarm-optimization-pso-an-overview/)
pages are useful API/equation checks. They do not determine this paper’s cleaning,
class merging, split, topology, search space, or metric aggregation.

## Full repeated-seed execution

The following example runs fixed published binary SSA parameters for the required
controlled seeds. It is expensive and is shown for explicit user-launched execution;
it was not run automatically:

```powershell
$seeds = @(42, 52, 62, 72, 82)
foreach ($seed in $seeds) {
  & $python -m ids_repro train `
    --cache-dir ".\cache\cicids2017" `
    --task binary --model cnn-lstm --swarm ssa `
    --protocol paper_replication --mode full `
    --seed $seed `
    --output-dir ".\results\cicids2017\binary\cnn-lstm\fixed_ssa\seed_$seed"
}
```

A paper-budget corrected optimizer search is exactly:

```powershell
& $python -m ids_repro search `
  --cache-dir ".\cache\cicids2017" `
  --task binary --model cnn-lstm --algorithm ssa `
  --mode full --fitness accuracy `
  --population-size 10 --iterations 100 `
  --seed 42 `
  --output ".\results\cicids2017\binary\cnn-lstm\ssa\seed_42\search_result.json"
```

Repeat with `--algorithm pso` and `--algorithm random` at the same budget, then use
each saved `best_parameters.json` in a fresh final `train --params-json` run. A
search result is not a test result.
