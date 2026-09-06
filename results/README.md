# Preserved experiment evidence

`results/` contains the compact, citable evidence used by the dissertation.
Raw cells and training checkpoints remain under `runs/` during execution. The
submission release directory archives the six audited checkpoints and their
main raw cells with checksums.

| Directory | Purpose |
|---|---|
| `dissertation_v10_corrected/` | Final corrected summaries, controls, release package, and explanation-audit decision |

The final frozen checkpoints, clean evaluations, and raw audit cells are in
`runs/dissertation_v10_corrected/`. Reproduction commands and the expected
outputs are documented in `docs/REPRODUCE_EXPERIMENTS.md`.

The final submission excludes superseded experiment outputs so that every
reported numeric result has one clear source.
