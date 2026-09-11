# Unified workstation rerun (v12)

The user authorized a complete rerun and subsequent thesis/figure update.
Protocol: configs/experiments/dissertation_v12_full_rerun.toml.
All three model families use seeds 42–46 and 50 epochs. The GAT pair shares
learning-rate decay and checkpoint-selection rules; MLP retains its declared
architecture-specific learning rate. Fixed demand splits and release thresholds
are unchanged. No old checkpoint or result is reused.

Run with `.venv/bin/python scripts/run_full_rerun.py`. The wrapper fixes CPU,
one OpenMP/MKL thread, TraCI, and a locally built SUMO 1.20.0 distribution. It records
hardware, code/config identity, commands, stage times and failure status.
Generated evidence stays under ignored runs/dissertation_v12_sumo120 so
later stages can verify a clean, frozen implementation revision.

Environment: Python 3.11, NumPy 1.26.4, Torch 2.2.2, SUMO/bindings 1.20.0.
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

## Runtime qualification failure

The initial SUMO 1.27.0 attempt at revision 27fdff0 stopped during MLP seed 45,
epoch 6, with UnicodeDecodeError while decoding getTaxiReservations. The initial
three completed MLP runs and one partial run remain in runs/dissertation_v12_full_rerun
as failed runtime-qualification evidence. None will be used in the thesis.
The full rerun switches to SUMO 1.20.0, matching the archived experiment runtime,
with matching Python bindings and a fresh output root. All 15 models restart.
This is a runtime compatibility decision made before held-out evaluation, not
selection based on model performance. The committed road/demand inputs remain
unchanged. The official 1.27.1 changelog did not identify a matching reservation
decoding fix: https://eclipse.dev/sumo/docs/ChangeLog.html.

The 1.20.0 arm64 wheel requires unavailable Homebrew 3.2 Xerces/FOX libraries.
The CLI binary is instead built from upstream tag v1_20_0, commit
96efa4d36d6c9af50710015bb725fc51730b582a, without source modifications:

```sh
cmake -S runs/runtime/sumo-1.20.0 -B runs/runtime/sumo-1.20.0/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DFOX_CONFIG= -DCHECK_OPTIONAL_LIBS=OFF -DENABLE_PYTHON_BINDINGS=OFF
cmake --build runs/runtime/sumo-1.20.0/build --target sumo --parallel 6
```

Build uses AppleClang 15.0.0.15000100, Xerces-C 3.3.0 and PROJ 9.7.0.
No GUI is used. The runner records the binary hash and upstream revision.
The pip SUMO binary is not on the selected runtime path.
