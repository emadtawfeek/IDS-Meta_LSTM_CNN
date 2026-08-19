# Status of the preserved author release

The untouched release was preserved at `../author-original` on the audit workstation
and is available from the authors' public GitLab repository. It is not vendored here
and is not treated as an executable reference implementation.

| File | Audit status |
|---|---|
| `CNN.py`, `CNN-PSO.py` | Byte-identical. Hard-coded missing CSV path, StandardScaler instead of the paper’s MinMaxScaler, and no real PSO in the nominal PSO copy. |
| `PSO-CNNLSTM.py` | Mixes decoded categorical/integer values into normalized-coordinate velocity equations, refits a label encoder on validation, and supplies rank-2 arrays to Conv1D. |
| `PSO-LSTM.py` | Same coordinate and label issues; additionally trains on a random half sample and passes rank-2 data to the final LSTM evaluation. |
| `SSA-CNN.py` | Defines an inconsistent SSA function, but the invoked optimizer is ten independent random trials; Conv1D input remains rank 2. |
| `SSA-CNNLSTM.py` | The invoked optimizer is random trials, not SSA; rank-2 Conv1D input and hard-coded nine-class output. |
| `SSA-LSTM.py` | Defined SSA reinitializes the population and has inconsistent optimization direction; invoked path is random trials. |

Common issues include hard-coded local paths, unavailable generated CSVs, mixed
`keras`/`tf.keras`, missing dependency versions, validation encoder refitting,
hard-coded class counts, absent saved seeds/models/traces, and no test safeguards.

The corrected implementation does not import these files. It transcribes the
published search space and architecture into typed, tested modules and labels fixed
paper hyperparameters separately from optimizer-derived parameters.
