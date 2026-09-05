# Preserved experiment evidence

`results/` contains the compact, citable evidence used by the dissertation.
Large raw cells and training checkpoints remain under the ignored `runs/`
directory.

| Directory | Purpose |
|---|---|
| `dissertation_v9_exposure_audit/` | Final validated summaries, controls, provenance checks, and explanation-release decision |
| `supporting_audits/` | Legal-random/greedy capability lower bounds and the request-node construct-validity check |

The final frozen checkpoints and clean evaluations are in
`runs/dissertation_v8/`. Raw exposure-conditioned audit cells are in
`runs/dissertation_v9_exposure_audit/`. Reproduction commands and the expected
outputs are documented in `docs/REPRODUCE_EXPERIMENTS.md`.

Do not compare retained supporting audits as if they were additional trained
model replications. They provide measurement and capability context for the
final v9 analysis.
