# Unified workstation rerun (v12)

The user authorized a complete rerun and subsequent thesis/figure update.
Protocol: configs/experiments/dissertation_v12_full_rerun.toml.
All three model families use seeds 42–46 and 50 epochs. The GAT pair shares
learning-rate decay and checkpoint-selection rules; MLP retains its declared
architecture-specific learning rate. Fixed demand splits and release thresholds
are unchanged. No old checkpoint or result is reused.

Run with `.venv/bin/python scripts/run_full_rerun.py`. The wrapper fixes CPU,
one OpenMP/MKL thread, TraCI, and the packaged SUMO distribution. It records
hardware, code/config identity, commands, stage times and failure status.
Generated evidence stays under ignored runs/dissertation_v12_full_rerun so
later stages can verify a clean, frozen implementation revision.

Environment: Python 3.11, NumPy 1.26.4, Torch 2.2.2, SUMO/bindings/data 1.27.0.
Exact installed versions are recorded in requirements-v12-lock.txt and manifests.
This is a new protocol on new hardware, not an exact historical reproduction.
The environment version parser supports both legacy 'Version' and the current
'Eclipse SUMO sumo' version banner; version equality checks remain enforced.

Stability failures stop the pipeline and must be retained and investigated,
not replaced with favorable seeds. Completed evidence must pass preflight.
After completion, regenerate numerical figures, inspect their annotations,
update all thesis numbers and conclusions, then render/verify Word and PDF.
Existing figure annotations and thesis claims are historical and must not be
copied into the new results without checking the computed evidence.
