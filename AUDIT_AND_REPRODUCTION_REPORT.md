# Audit and reproduction report

## Scope and preservation statement

This report covers the attached 2025 paper, the public PNRAI author release, every
Python/configuration/test file in the submitted reproduction, the provided
CIC-IDS2017 and NSL-KDD dataset trees, both prepared caches, every saved model/result
directory, and the independently regenerated artifacts under `results/`.

The audit was read-only until the inventory was complete. No raw dataset, archive,
paper PDF, author source file, prepared cache, saved model, or earlier output was
deleted or overwritten. The author release remains at `../author-original` on the
audit workstation and is not vendored in this GitHub repository. The original full
and smoke artifacts remain under the workstation's `outputs/`. Corrections were
made in `paper-reproduction/src`; lightweight committed evidence is under
`evidence/`, while bulky generated artifacts remain local under `results/`.

The source paper SHA-256 is
`d3ff03cf8648d33c2d6807dfa106e7a92a502c06382cd62a40c6a90f2a2a7f5f`.
The preserved author clone is at commit `bf90ddc` (2025-08-21).

### Complete file inventory and roles

The audit-workstation file `results/audit/FILE_INVENTORY.json` is the machine-readable inventory of every
audited non-environment file under the CIC project and NSL-KDD source tree. It
records absolute path, byte size, SHA-256, and one role for each file. Virtual
environment packages, Git internals, Python bytecode, and pytest caches are excluded
because they are runtime/tool state rather than submitted project artifacts. The
final inventory contains 291 source, dataset, cache, model, output, test,
configuration, documentation, and generated-evidence files.

That absolute-path inventory is intentionally not committed to GitHub. The portable,
text-only numerical evidence is committed under `evidence/`.

| File family | Role |
|---|---|
| `s11227-025-07802-w.pdf` | Source experimental paper |
| `author-original/*.py` | Preserved legacy author implementations |
| `MachineLearningCSV/**`, `GeneratedLabelledFlows/**` | CIC source data and archives |
| `dataset/nsl-kdd/**` | NSL-KDD source tables, ARFF files, and supplied documentation |
| `cache/**` | Prepared arrays, fitted preprocessors/encoders, mappings, split indices, manifests |
| `outputs/**` | Preserved pre-audit models, predictions, histories, reports, and metrics |
| `src/ids_repro/**` | Corrected implementation and modular public API |
| `configs/**` | Paper/rigorous YAML run contracts |
| `tests/**` | Automated correctness and regression tests |
| `scripts/**` | Inventory and repeated-run summarization utilities |
| `results/**` | New non-destructive audits, generated tables, smokes, and search traces |
| `README.md`, this report, `LEGACY_IMPLEMENTATIONS.md`, `CHANGE_MANIFEST.md` | Project-specific documentation and change record |

The exact created/modified file list is in `CHANGE_MANIFEST.md`; individual roles
and hashes are intentionally kept in JSON so they can be checked automatically.

### Saved-artifact provenance

| Artifact family | Determinable generator | Status |
|---|---|---|
| `outputs/cic-binary-cnn-lstm-ssa` | Earlier `ids_repro.training.train_and_evaluate` fixed Table 6 SSA preset path | Complete 79-epoch fixed-parameter binary run; not an SSA search |
| `outputs/smoke-cic-binary-ssa`, `outputs/smoke-test` | Earlier fixed-preset `train` command with sample/epoch caps | Byte-identical CIC feasibility outputs |
| `outputs/smoke-nsl-multiclass-ssa` | Earlier fixed-preset NSL `train` command | Legacy-cache extension smoke |
| `outputs/pso-smoke.json` | Earlier corrected `particle_swarm_search` CLI path | Bounded validation search JSON only |
| `results/audit/submitted-cic-binary-ssa` | `ids_repro.reporting.audit_saved_run` | Independent recalculation from preserved predictions/cache truth |
| `results/paper-recalculation-final` | `ids_repro.reporting.write_paper_tables` | Metrics generated from published matrices and submitted audit JSON |
| `results/cicids2017/**/seed_42_smoke` | Current corrected train/search CLI paths | Explicit smoke artifacts, never full results |
| `results/nsl-kdd/**/seed_42_smoke` | Current corrected trainer on newly prepared rigorous cache | Dataset-extension smoke, independently audited |

For preserved author scripts, no saved artifact embeds enough provenance to identify
a particular legacy filename. The full submitted CIC artifact is determinable as the
corrected fixed-preset trainer because its JSON schema, exact model topology/input,
parameter count, hyperparameter payload, and 79-epoch history all match that path.

## Executive findings

1. **The CIC row population and paper test split are exactly reproducible.** The
   missing operations are invalid-row removal followed by exact-row deduplication,
   alphabetic input-file order, then two unstratified `train_test_split` calls with
   seed 42. The reconstructed 504,160-row test class counts match every published
   confusion-matrix row total.
2. **The released PSO and SSA code is not a valid executable implementation of the
   claimed search.** PSO mixes decoded parameter values with normalized coordinates.
   The invoked SSA CNN/CNN-LSTM/LSTM paths are random trials; other defined SSA
   functions reinitialize populations or use inconsistent objective direction.
3. **The earlier full saved model is real and internally consistent, but it is a
   fixed-hyperparameter run.** Its saved predictions independently reproduce
   `[[418246,652],[327,84935]]` and 99.8058156% accuracy. It does not prove that an
   SSA search produced those parameters.
4. **The saved model is close to, but not the same as, the paper’s binary SSA
   matrix.** It has 145 additional false positives, 28 fewer false negatives, and
   117 more total errors than the paper matrix.
5. **The paper’s multiclass headline metrics mix aggregation schemes.** Its stated
   precision is approximately macro precision, while stated recall/F1 are close to
   micro values. Standard metrics independently calculated from Figure 9 are
   reported below.
6. **The paper’s CNN/LSTM use is not temporal.** In replication mode, Conv1D treats
   feature columns as an ordered axis and LSTM sees a one-step vector. Corrected code
   names this `feature_axis_replication`. It must not be described as learning packet
   or flow sequences.
7. **NSL-KDD has no paper target.** It is supported as a separate extension using
   the official train/test split, a train-only stratified validation holdout, and a
   train-fitted numeric/categorical transformer.

## Paper-to-code reconstruction

### Paper claims that can be operationalized

| Item | Paper/recovered specification | Reproduction status |
|---|---|---|
| Dataset | CIC-IDS2017 MachineLearningCVE | Exact supplied files identified |
| Classes | Binary and nine grouped classes | Exact mapping implemented |
| Cleaning | Invalid values removed; eight empty features removed | Implemented; exact dedup inferred from matrices |
| Scaling | Min-max | Fit on training rows only |
| Split | 70% train, 10% validation, 20% test | Exact paper-compatible indices recovered |
| Architectures | CNN, LSTM, CNN-LSTM | Implemented with explicit rank-3 assertions |
| Search space | Tables 4/5 and reported presets Tables 6/7 | Typed normalized search space and exact presets |
| PSO/SSA budget | population 10, 100 iterations | Supported; full 1,000-fit searches not launched |
| Final paper weights | Not released | Not exactly reproducible |

### Paper inconsistencies

- The abstract reports binary SSA CNN-LSTM accuracy of 99.83%; the conclusion says
  99.80%.
- The abstract/results report multiclass accuracy around 99.7%; the conclusion says
  97.00%.
- Table 2 reports an impossible Bot percentage and 15,070 Web Attack Brute Force
  rows; the raw file and the paper’s test matrix imply 1,507 raw rows.
- Table 8 reports SSA Bot recall of 39%, but the exact matrix has 176/391 = 45.0128%.
- Figure 8’s approximately 97.0% macro precision and 87.3% macro recall are
  compatible with Figure 9, but the standard arithmetic mean of the nine per-class
  F1 scores is 90.7379%, not the 99.7% headline.
- Table 9 combines multiclass accuracy/micro recall/micro F1 near 99.7% with macro
  precision near 97.3%. Those values are not a single averaging protocol.
- The SSA pseudocode branches on `c3 <= 0` although `c3` is sampled in `[0,1]`.
  Corrected code uses the canonical operational split `c3 < 0.5` and records it.

### Independently recalculated paper metrics

All values below are generated from the transcribed published confusion matrices by
`ids_repro.reporting`, not copied from paper summary tables.

| Task | Matrix | Accuracy | Balanced/macro recall | Macro precision | Standard macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| Binary | PSO | 0.998054 | 0.996657 | 0.996421 | 0.996539 | 0.998054 |
| Binary | SSA | 0.998290 | 0.997313 | 0.996608 | 0.996960 | 0.998291 |
| Multiclass | PSO | 0.997509 | 0.867502 | 0.942607 | 0.890135 | 0.997408 |
| Multiclass | SSA | 0.997366 | 0.873520 | 0.970281 | 0.907379 | 0.997297 |

Generated JSON, CSV, and Markdown sources are committed under
`evidence/paper-recalculation/`, including the submitted-versus-paper table.

## CIC-IDS2017 forensic audit

### Schema and row cleaning

The eight MachineLearningCVE CSVs contain 2,830,743 rows and 79 columns: 78 numeric
features plus `Label`. The combined audit found:

| Stage | Rows/cells |
|---|---:|
| Raw rows | 2,830,743 |
| Rows with missing/nonnumeric feature values | 1,358 |
| Missing cells | 1,358 |
| Rows with positive/negative infinity | 2,867 |
| Infinite cells | 4,376 |
| Union of invalid rows removed | 2,867 |
| Exact duplicate rows after invalid removal | 307,078 |
| Final clean unique rows | 2,520,798 |

Per-file counts establish the source ordering and cleaning provenance:

| Alphabetic source file | Raw | Invalid | Duplicate | Clean unique |
|---|---:|---:|---:|---:|
| Friday Afternoon DDoS | 225,745 | 34 | 2,629 | 223,082 |
| Friday Afternoon PortScan | 286,467 | 371 | 73,842 | 212,254 |
| Friday Morning | 191,033 | 122 | 10,814 | 180,097 |
| Monday | 529,918 | 437 | 27,017 | 502,464 |
| Thursday Afternoon Infiltration | 288,602 | 207 | 42,539 | 245,856 |
| Thursday Morning Web Attacks | 170,366 | 135 | 13,652 | 156,579 |
| Tuesday | 445,909 | 264 | 35,662 | 409,983 |
| Wednesday | 692,703 | 1,297 | 100,923 | 590,483 |

The only eight features constant after cleaning/deduplication are exactly:

`Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Avg Bytes/Bulk`,
`Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`,
`Bwd Avg Packets/Bulk`, and `Bwd Avg Bulk Rate`.

### Labels

Raw counts include 2,273,097 BENIGN, 128,027 DDoS, 231,073 DoS Hulk,
158,930 PortScan, 10,293 GoldenEye, 5,499 Slowhttptest, 5,796 Slowloris,
7,938 FTP-Patator, 5,897 SSH-Patator, 1,966 Bot, 1,507 Web Brute Force,
652 XSS, 36 Infiltration, 21 SQL Injection, and 11 Heartbleed rows.

After cleaning/deduplication and the paper’s nine-class grouping, the recovered test
counts are:

`[418898, 391, 1754, 25502, 39014, 3, 6, 18155, 437]`

for BENIGN, Bot, Brute Force, DDoS, DoS, Heartbleed, Infiltration, PortScan,
and Web Attack. Their total is 504,160 and matches both paper multiclass matrices.

### Exact split reconstruction

The paper-compatible path is:

1. read the eight files in alphabetic filename order;
2. convert feature columns to numeric and remove any row with NaN/non-numeric/inf;
3. remove exact duplicate rows while keeping the first occurrence;
4. remove the eight constant features;
5. map 15 raw strings to nine classes;
6. split 80/20 with `random_state=42`, shuffle, no stratification;
7. split 12.5% of the 80% for validation with the same seed and behavior;
8. fit MinMaxScaler on training indices only and transform validation/test.

This yields 1,764,558 training, 252,080 validation, and 504,160 test rows. The
unstratified split is retained only as `paper_replication`; the
`rigorous_evaluation` protocol
stratifies both calls. New preparations save all split indices. The preserved old
cache predates that enhancement and is not rewritten.

### Source hashes

| File abbreviation | SHA-256 |
|---|---|
| Friday DDoS | `6ff1580f5f81c0ae28a26f7631721018577f5f7c5e0feac28b795fcfe7b411ee` |
| Friday PortScan | `ca1824c51bfbb7b3c72290a11be04366ba8815878c6a1cc5c44cb1cee269e99b` |
| Friday Morning | `53a41c24d570ea83b7ac55b2e94df94e7a8216aeb80a2af0246b6bc8bb543000` |
| Monday | `852c4beb34eda186f32561fa79df7a0747e92e1a6535b01270820dd9ffe17f34` |
| Thursday Infiltration | `6bcda3857c2504676034e3ea57762d38393cc734cb377a726bd5cb153961b1b5` |
| Thursday Web | `d67066211fb1689c78406f1506f4c44704ecb92088353d5c96d96d6474eb819d` |
| Tuesday | `52b8692ae8c7d2ed04671fe2b98335693c0a92c7ab157d8c8b534d6523080851` |
| Wednesday | `893c27dc968bf7a8adef1689f90be55ca4a4dc3088fb63d6ff247ac56856df2a` |

## NSL-KDD audit and extension protocol

`KDDTrain+.txt` has 125,973 rows and `KDDTest+.txt` has 22,544 rows. Each has
41 features, a raw label, and difficulty field. `protocol_type`, `service`, and
`flag` are categorical. No missing, infinite, or exact duplicate rows were found.
`num_outbound_cmds` is constant in the training file.

Grouped official-train counts are normal 67,343; DoS 45,927; Probe 11,656;
R2L 995; U2R 52. Official-test counts are normal 9,711; DoS 7,460; Probe 2,421;
R2L 2,885; U2R 67.

The `rigorous_evaluation` adapter reserves the official test file, stratifies a 12.5% validation
holdout from official training, fits MinMaxScaler and OneHotEncoder on the remaining
training rows only, handles unknown categories, produces 122 transformed features,
and saves the fitted ColumnTransformer and mapping. Key hashes are:

- `KDDTrain+.txt`: `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95`
- `KDDTest+.txt`: `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84`

NSL-KDD results must be reported as an extension and compared only with other runs
using this documented protocol—not with the paper’s CIC numbers.

## Temporal and tensor semantics

MachineLearningCVE has no timestamp, flow/session ID, or capture-order field.
NSL-KDD also lacks timestamps and stable session identifiers. Therefore the prepared
adapters explicitly disable `temporal_window` with a clear error.

The separate CIC `GeneratedLabelledFlows` exports do contain Flow ID, endpoint
fields, protocol, Timestamp, 78 statistical features, and Label. Timestamp formats
vary by capture. The corrected protocol utilities support stable time sorting,
configurable `window_size` and `stride`, last/majority labeling, construction inside
one capture/session group, chronological within-group 70/10/20 splitting, and
disjointness assertions. They were tested synthetically. A full GeneratedFlows
temporal cache was not built in this audit, so no temporal performance claim is made.

The replication models use explicit input contracts:

| Mode/model | Tensor | Meaning |
|---|---|---|
| Feature-axis CNN/CNN-LSTM | `(N, feature_count, 1)` | Feature columns are the Conv1D axis |
| Feature-axis LSTM | `(N, 1, feature_count)` | One artificial time step |
| Temporal CNN/LSTM/CNN-LSTM | `(N, time_steps, feature_count)` | Real ordered windows only |

Runtime checks reject rank-2 Conv1D/LSTM input, invalid kernels/pools, one-step
“temporal” data, and temporal requests on unsupported caches.

## Legacy implementation audit

The detailed file table is in `LEGACY_IMPLEMENTATIONS.md`. Principal failures are:

- hard-coded absent CSV paths and unavailable generated datasets;
- validation LabelEncoder refitting, which can silently remap classes;
- StandardScaler in a binary script despite the paper specifying min-max scaling;
- random half-dataset sampling in PSO-LSTM;
- missing Conv1D/LSTM rank-3 reshapes;
- hard-coded nine outputs without asserting observed classes;
- PSO coordinate arithmetic between normalized positions and decoded values;
- SSA functions that minimize in one place and maximize in another, reinitialize
  the population, or are never called;
- invoked “SSA” optimization implemented as ten random trials;
- no dependency lock, seed record, model/search trace, probabilities, or split IDs.

The release is valuable for topology/search-space clues but cannot be executed as
published evidence of the claimed optimization.

## Corrected implementation

The corrected project provides:

- typed paper presets and YAML run contracts with configurable paths;
- separate `paper_replication` and `rigorous_evaluation` split policies;
- train-only preprocessing with serialized transformers, label maps, feature names,
  source hashes, class counts, and new split indices;
- explicit feature-axis versus temporal semantics;
- CNN, LSTM, and CNN-LSTM output dimensions derived from the task/classes;
- normalized-coordinate PSO, canonical operational SSA, and budget-matched random
  search, all maximizing one documented validation fitness;
- accuracy, macro-F1, or cost-sensitive validation objectives without test access;
- full candidate, position, convergence, timing, seed, and decoded-parameter traces;
- final test evaluation only after parameter choice;
- binary specificity/FAR/FNR plus macro/weighted metrics and ROC-AUC/PR-AUC when
  probabilities are available;
- multiclass macro/micro/weighted precision/recall/F1, balanced accuracy, support,
  confusion matrix, and probability-based macro AUCs;
- saved truth, predictions, probabilities, threshold policy, configuration, model,
  model summary/count, history, best validation epoch, timings/resources,
  environment, dataset manifest, and subset indices;
- repeated-run mean/std/95% CI and paired test/effect-size utilities.

Corrected run writers refuse to overwrite a non-empty output directory. Search
writers likewise refuse an existing result file.

## Saved full-run audit

Source: preserved `outputs/cic-binary-cnn-lstm-ssa`.

The model has input `(None,70,1)`, binary output, 232,833 parameters, and layers
Conv1D → MaxPool1D → LSTM → two Dense/Dropout blocks → sigmoid. Its history contains
79 epochs and therefore matches the fixed Table 6 SSA epoch count. It was produced
by the earlier fixed-preset training command, not a search command.

Saved `predictions.npy` SHA-256:
`377d999dce845ef02a6b8e1f2f7140ab99d8ce9cf5b3ec0207204534e2702370`.

Independent recalculation from the cache test truth gives:

| Metric | Saved run | Paper binary SSA matrix | Difference (percentage points where applicable) |
|---|---:|---:|---:|
| Accuracy | 0.998058156 | 0.998290225 | -0.023207 pp |
| Balanced accuracy/macro recall | 0.997304149 | 0.997313022 | -0.000887 pp |
| Macro precision | 0.995800397 | 0.996608218 | -0.080782 pp |
| Macro F1 | 0.996550399 | 0.996960209 | -0.040981 pp |
| Attack precision | 0.992382021 | 0.994064205 | -0.168218 pp |
| Attack recall | 0.996164763 | 0.995836363 | +0.032840 pp |

Confusion comparison:

| Source | TN | FP | FN | TP | Errors |
|---|---:|---:|---:|---:|---:|
| Saved full run | 418,246 | 652 | 327 | 84,935 | 979 |
| Paper SSA | 418,391 | 507 | 355 | 84,907 | 862 |
| Saved − paper | -145 | +145 | -28 | +28 | +117 |

All eight originally saved scalar metrics match independent recalculation. ROC-AUC
and PR-AUC cannot be recovered because the earlier run saved only hard predictions,
not probabilities. The new trainer saves probabilities. The independent audit is
in `results/audit/submitted-cic-binary-ssa/`; it does not modify the source run.

## Verification performed

### Automated tests

`python -m pytest` passes **35 tests** in 12.76 seconds. The 16 emitted warnings are
third-party TensorFlow/protobuf and Matplotlib deprecations; there are no test
failures. Coverage targets include:

- exact paper presets and paper test total;
- binary positive-class metrics and probability-based AUC;
- standard macro/micro/weighted paper-matrix aggregation;
- disjoint, complete, repeatable and stratified split behavior;
- time ordering and no cross-group temporal windows;
- feature-axis and temporal tensor shapes for CNN/LSTM/CNN-LSTM;
- invalid temporal/kernel inputs;
- PSO/SSA budgets, normalized state, decoding, tracing, and random baseline;
- repeated-run confidence intervals and paired-test metadata.
- cache/protocol mismatch rejection before model fitting.

Both final cache verification commands passed finite/shape checks. Held-out
MinMax-transformed values can legitimately exceed 1.0 (CIC validation max 1.3495,
CIC test max 1.3271, NSL test max 2.5) because the scaler is correctly fit only on
training rows; clipping or refitting held-out data would alter the experiment.

### Bounded execution smoke

A new one-epoch, 20,000/5,000/5,000-row paper-replication CNN-LSTM path completed at
`results/cicids2017/binary/cnn-lstm/fixed_ssa/seed_42_smoke`. It generated all
corrected artifacts and obtained 91.54% accuracy; this is a feasibility value, not a
scientific result, because the sample and epoch budget are deliberately tiny.

A separate one-epoch NSL-KDD extension smoke used a newly prepared stratified
`rigorous_evaluation` cache and completed at
`results/nsl-kdd/multiclass/cnn-lstm/fixed_ssa/seed_42_smoke`. Its independently
verified accuracy is 69.04% on the deterministic 5,000-row smoke subset. It is not a
paper result and not evidence about full-training performance.

PSO, SSA, and random search each completed two iterations with two CNN candidates
per iteration, one epoch, 500 train rows, and 200 validation rows. Structured traces,
candidate seeds/runtimes, normalized position histories, configuration, fitted
preprocessor, and convergence plots are under
`results/cicids2017/binary/cnn/{pso,ssa,random}/seed_42_smoke/`. These tiny imbalanced
subsets produced tied macro-F1 values; they verify execution, population persistence,
and accounting only, not optimizer quality.

The earlier development artifact `results/smoke/cic-binary-corrected` was produced
before the cache/protocol mismatch guard and labeled a paper-split cache as rigorous.
It is preserved for provenance but is invalid for protocol comparison and excluded
from all result tables. The trainer now rejects that mismatch.

### Intentionally not executed

- the paper-scale 10 × 100 search for each optimizer/model/task;
- five-or-more-seed full fixed-parameter or optimizer-derived experiments;
- full corrected multiclass CIC training;
- full corrected NSL-KDD training/search;
- a full timestamped GeneratedLabelledFlows temporal experiment;
- significance tests on full runs (only one full submitted seed exists).

No claim is made for any of these unexecuted experiments.

## Reproduction status and next experiments

| Claim | Status |
|---|---|
| CIC source population | Reproduced exactly |
| CIC cleaning/dedup and paper test population | Reproduced exactly |
| Published test row totals | Reproduced exactly |
| Published fixed hyperparameter tables | Transcribed and tested |
| Saved binary fixed-preset model | Independently verified, close but not identical |
| Author PSO/SSA execution | Not valid in legacy code; corrected implementations smoke-tested |
| Paper-scale optimizer search | Not run |
| Exact neural weights | Impossible from released materials |
| NSL-KDD paper result | Not applicable; extension only |
| Temporal result | Not run; row-level paper mode is non-temporal |

Recommended scientific sequence:

1. prepare new protocol-specific caches rather than rewriting the preserved cache;
2. run fixed CNN, LSTM, and CNN-LSTM baselines for seeds 42, 52, 62, 72, and 82;
3. run budget-matched random, PSO, and SSA searches on the same validation data;
4. train one final model per chosen candidate/seed and touch test data once;
5. aggregate mean, sample standard deviation, 95% CI, paired test, and effect size;
6. report `paper_replication` and `rigorous_evaluation` protocols in separate tables;
7. report NSL-KDD in a separate extension table;
8. only add a temporal table after constructing and validating the GeneratedFlows
   adapter and documenting its grouping/window label policy.

## 2026-08-19 reproducibility correction addendum

The implementation was re-audited from baseline commit
`441ba6b7e4293af9eab155d79b2d2417557ec2eb`. The detailed additive change log is
`CORRECTION_MANIFEST_2026-08-19.md`.

The corrected search objective is now evaluated after every epoch and uses the
same requested fitness, patience, and `min_delta` rules for PSO, SSA, and the
random baseline. Search and rigorous final training restore the selected epoch's
weights. Fixed paper-protocol runs retain their endpoint weights while recording
the best validation epoch, so fixed-epoch paper comparison is not silently changed.

Capped samples are now deterministic stratified row subsets with saved class
counts, indices, and checksums. CIC `rigorous_evaluation` remains only a stratified
random row split: it is not leakage-free, group-independent, capture-independent,
session-independent, or time-independent. Temporal config fields remain
unavailable until a complete timestamp/session-aware adapter is implemented.

Search checkpoint/resume now persists optimizer population/dynamics, best states,
iteration/member counters, evaluation trace, candidate cache, and RNG state after
every completed candidate. The trace distinguishes proposals, unique decoded
configurations, cache hits, neural-fit attempts, completed fits, and failures.
Parameter provenance, strict paper-comparison eligibility, expanded metrics and
latency/memory reporting, and prediction-level McNemar comparison were also added.

The matched PSO/SSA/random run recorded in
`evidence/review-2026-08-19/matched_smoke_summary.json` is a two-candidate,
one-epoch engineering smoke only. It is not a scientific optimizer comparison or
a paper reproduction result.

Exact PowerShell commands and artifact descriptions are in `README.md`.
