# Reproducibility correction manifest — 2026-08-19

Baseline: `441ba6b7e4293af9eab155d79b2d2417557ec2eb` on `main`.
This correction is additive: legacy code/evidence was not deleted or rewritten.

## Resolved in this correction

- A single validation objective is evaluated after every epoch for candidate
  search and final training. Best epoch/fitness, patience, `min_delta`, and full
  history are saved. Searches and rigorous final runs restore best weights; fixed
  paper-protocol runs retain endpoint weights for the stated fixed-epoch protocol.
- Sample caps are deterministic and label-stratified. They fail if the cap cannot
  represent every observed class and save indices, class counts, and checksums.
- Parameter selection is explicitly labeled as `paper_preset`, `pso_search`,
  `ssa_search`, `random_search`, `manual`, or `transferred_cic_preset`. Search
  artifacts are validated against algorithm, seed, fitness, budget, dataset,
  task, model, cache/split identity, raw position, and decoded parameters.
- Paper confusion comparison is gated to a complete CIC-IDS2017 paper-protocol
  fixed paper-preset run with exact parameters, class order, and saved split
  checksum. Ineligible runs save the reasons and no comparison.
- PSO, SSA, and random search atomically checkpoint every completed proposal,
  including optimizer/RNG/objective-cache state. Resume validates the complete
  contract and continues without skipping or repeating a checkpointed proposal.
- Decoded duplicates are cached by parameters plus dataset/split/subset/task/model/
  seed/epoch-cap/fitness identity. Search traces separate proposals, unique
  configurations, cache hits, fit attempts, completed fits, and failures.
- Invalid/non-finite candidates are recorded and isolated. PSO supports validated
  velocity clipping. TensorFlow cleanup is in a `finally` block.
- Cost-sensitive fitness is currently binary-only and validates finite,
  non-negative, nonzero coefficients. Its normalized FAR/FNR equation and
  coefficients are persisted.
- Evaluation adds MCC, binary micro precision/recall/F1, predicted counts,
  serialized model size, warmed-batch latency, throughput, per-sample inference
  time, and periodically sampled peak RSS.
- Prediction-level McNemar testing validates identical truth arrays and split
  checksums and reports both discordant directions with exact small-sample testing.
- CIC rigorous evaluation is documented as a stratified random row split, not a
  leakage-free or time-independent protocol. Temporal config fields are explicitly
  unavailable until a complete timestamp/session-aware adapter exists.

## Verification actually performed

- `python -m compileall -q src tests`: passed.
- `python -m pytest -q`: 51 passed on Python 3.12.13 / TensorFlow 2.16.2 CPU.
- A bounded checkpoint/resume neural test ran two one-epoch fits and passed.
- Matched CIC binary CNN-LSTM smoke searches ran for PSO, SSA, and random search
  with seed 42, macro-F1, population 2, one iteration, one epoch per candidate,
  2,000 training rows, and 1,000 validation rows. Each completed two neural fits
  with no failures. See `evidence/review-2026-08-19/matched_smoke_summary.json`.
- One capped final-training feasibility run loaded the PSO search JSON through the
  provenance validator and produced the expanded artifacts.

These are engineering feasibility checks, not paper reproduction results and not
evidence that one optimizer outperforms another.

## Deliberately unresolved / not claimed

- No 1,000-candidate paper-budget search was run.
- No five-seed full CIC or NSL-KDD experiment was run.
- No claim is made that the paper's published metrics were reproduced.
- The legacy CIC cache lacks persisted split-index files. Its paper split identity
  can be deterministically reconstructed for cache/search identity, but strict
  paper-confusion eligibility remains false until a new cache is prepared with
  saved indices and metadata checksum.
- Temporal/session-independent evaluation remains unavailable for the provided
  MachineLearningCVE and NSL-KDD adapters.
