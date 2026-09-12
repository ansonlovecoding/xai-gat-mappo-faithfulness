# Reproducing the current dissertation results

The thesis uses the completed v12 full rerun: five training seeds (42–46), three model families and 50 epochs per run. The source experiment revision is `2c604e7`. All final estimates come from the SUMO 1.20.0 run; earlier workstation results and the failed SUMO 1.27.0 qualification run are excluded.

- Protocol: `configs/experiments/dissertation_v12_full_rerun.toml`
- Runtime and source-build instructions: `docs/FULL_RERUN_V12.md`
- Raw working output: `runs/dissertation_v12_sumo120/`
- Version-controlled summaries and release evidence: `results/dissertation_v12_full_rerun/`
- Historical procedure: `docs/REPRODUCE_V10_HISTORICAL.md`

## Environment and execution

Use Python 3.11.14 with `requirements-v12-lock.txt`, install this project with `pip install -e . --no-deps`, and build upstream SUMO 1.20.0 as documented in FULL_RERUN_V12.md. The M3 Pro run used CPU execution, one Torch intra-op thread, default 12 inter-op threads and TraCI. The wrapper selects the built SUMO binary, not the incompatible arm64 wheel binary.

Run from a clean, committed checkout with a fresh output directory. Existing cells are immutable and must not be mixed with another configuration or revision.

```sh
.venv/bin/python scripts/run_full_rerun.py
```

The wrapper trains, selects, evaluates, runs deterministic diagnostics, main sweeps, random-loss sensitivity and attribution controls, then performs preflight, statistical analysis, explanation audit, matched baselines, precision sensitivity, release packaging and supplementary no-op analysis. Stability and preflight failures stop execution. The recorded run used additional independent workers for disjoint sweep/control directories; their exact commands and timing records are retained under `execution_records/`. Reproduction may run sequentially without changing the simulation-time protocol.

Expected completed work: 15 training/selection records; 120 clean evaluation cells (720 episodes); ten deterministic diagnostics (30 episodes); 400 main-sweep cells (2,400 episodes); 240 random-loss cells (720 episodes); 120 attribution/query-row cells (360 episodes); and 96 baseline episodes. There are ten full-precision and ten rounded analyses.

The original source revision precedes the addition of the wrapper's precision stage. At that revision, run `analyze_hypotheses.py` for each sweep with `--input-round-decimals 4 --output <sweep>/analysis_rounded4.json`, followed by `summarize_precision_sensitivity.py`. The archived precision orchestration script records the exact executed commands.

## Figures and dissertation

```sh
.venv/bin/python scripts/plot_dissertation.py --root runs/dissertation_v12_sumo120 --analysis-root results/dissertation_v12_full_rerun --out docs/figures --prefix v12
.venv/bin/python scripts/analyze_faithfulness_controls.py runs/dissertation_v12_sumo120/faithfulness_controls/B2_full runs/dissertation_v12_sumo120/faithfulness_controls/H5_full runs/dissertation_v12_sumo120/faithfulness_controls/B2_action_row runs/dissertation_v12_sumo120/faithfulness_controls/H5_action_row --out results/dissertation_v12_full_rerun/faithfulness_controls --fig-dir docs/figures --fig-prefix v12
.venv/bin/python scripts/plot_dataset_eda.py
.venv/bin/python scripts/plot_thesis_methodology.py
.venv/bin/python scripts/plot_main_findings_minimal.py
python scripts/build_dmu_thesis_docx.py
```

The DOCX builder also needs python-docx and its document-rendering dependencies. Render the document, update the page map with `scripts/update_thesis_page_map.py`, rebuild and render until the page map is stable. Check figure labels, table continuations and references visually.

`release/manifest.json` hashes selected checkpoints, training/validation records, raw compressed cells, analyses, configuration and execution records. Retain archived source paths as provenance; local reproduction paths can differ. Recompute all interpretations from the raw evidence rather than copying historical annotations.

## Interpretation boundaries

GAT and GAT-Outage have matched budgets and optimizer settings but different validation telemetry. This compares training-and-selection configurations, not training telemetry alone. The demand collection was used in pilot work, so the rerun is not validation on an independently unseen dataset. Confidence intervals describe episode-level variation within fixed checkpoints; five training seeds are summarized descriptively.
