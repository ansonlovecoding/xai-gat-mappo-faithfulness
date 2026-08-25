# Experiment codebase and rerun contract

## Purpose

This repository tests whether graph attention remains a trustworthy explanation
when fleet telemetry becomes stale. SUMO remains the ground-truth simulator.
Tunnel entry triggers signal loss, while `DegradationLayer` freezes the last
valid reading at the observation boundary. The policy therefore receives stale
data without changing the simulated vehicle state.

The definitive rerun is described by
`configs/experiments/dissertation_v3.toml`. Treat that file as the experiment
contract. Changes to seeds, demand splits, model definitions, degradation, or
faithfulness settings create a different experiment.

## Research endpoints

The primary endpoint is `stale_attention_shift`: the difference between the
absolute attention mass assigned to stale nodes in the degraded observation
and the mass assigned to the same slots in its exact clean twin. It uses only
a binary stale/fresh mask. A positive value means attention moved toward stale
nodes; it does not mean that a larger AoI value caused lower faithfulness.

Supporting measurements are:

- `stale_attention_mass`: total graph attention assigned to stale vehicles;
- `stale_attention_share`: stale-node share within vehicle attention;
- `stale_in_top3`: whether a stale vehicle appears among the top three nodes;
- attention drift: Jensen-Shannon divergence between degraded and clean twins;
- type-matched DEF: action-level explanation faithfulness;
- chosen-action exclusion DEF: construct-validity audit;
- WAMSN: secondary, AoI-weighted description of stale attention exposure;
- pickups, reward and waiting time: operating context, not the research target.

## Code responsibilities

```text
src/dispatch_marl/
  degradation.py              observation-boundary freeze mechanism
  env.py                      SUMO/PettingZoo environment and clean twins
  models/                     GAT, MLP and decoupled explanation models
  training.py                 rollout, GAE and PPO update
  faithfulness.py             attention, DEF and stale-attention metrics
  reproducibility.py          complete RNG setup and derived seeds
  runtime.py                  shared device and checkpoint loading
  provenance.py               hashes, runtime inventory and manifests
  experiment_validation.py    pre-analysis scientific gates

configs/experiments/
  dissertation_v3.toml        fixed model matrix and rerun protocol

scripts/
  train.py                    single immutable training run
  sweep_severity.py           single checkpoint severity sweep
  select_checkpoint.py        validation-only checkpoint selection
  preflight_experiment.py     result and manipulation validation
  analyze_hypotheses.py       cluster-aware statistical analysis
  summarize_dissertation_experiment.py  accepted cross-seed thesis tables
  run_dissertation_experiments.py  configuration-driven orchestration

tests/                        fast deterministic unit tests
runs/                         ignored working artifacts
results/                      frozen, citable experiment bundles
```

## Reproducibility rules

1. Run only from a clean, committed Git revision.
2. Match the SUMO binary, `traci`, and `sumolib` versions exactly.
3. Keep NumPy on the 1.26 line for the pinned PyTorch 2.2.2 wheel.
4. Select `ckpt_selected.pt` on validation demand only. The declared rule uses
   mean pickups, mean reward as a tie-breaker, then the earlier epoch. The test
   split is reserved for final evaluation and does not influence selection.
   Validation uses the final stochastic policy mode with a fixed RNG seed.
   Selection must also pass the declared minimum pickup and improvement gates.
5. Do not reuse an output directory for a changed configuration. The manifest
   hash rejects this, while compatible incomplete sweeps may resume by cell.
6. Keep every raw cell JSON. Statistical results without their raw records are
   not a complete citable artifact.
7. Run preflight before analysis. A failed result must not enter thesis tables.

## Commands

Check the environment and tests:

```bash
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/test_policy.py
.venv/bin/python scripts/test_faithfulness.py
```

Inspect every command without starting a long run:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py --dry-run
```

Run one model as a pilot:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py --stage train --model B2_gat
.venv/bin/python scripts/run_dissertation_experiments.py --stage select --model B2_gat
.venv/bin/python scripts/run_dissertation_experiments.py --stage sweep --model B2_gat
.venv/bin/python scripts/run_dissertation_experiments.py --stage preflight --model B2_gat
.venv/bin/python scripts/run_dissertation_experiments.py --stage analyze --model B2_gat
```

Use `--resume` only to skip complete training runs and cells belonging to the
same manifest hash. Run all configured stages after the pilot has passed:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py --resume
```

## Artifact structure

```text
runs/dissertation_v3/
  training/<model>/seed_<training-seed>/
    manifest.json
    args.json
    train_log.jsonl
    ckpt_final.pt
    ckpt_selected.pt
    checkpoint_selection.json
    validation/*.json
  evaluations/<model>/seed_<training-seed>/
    eval_seed_<evaluation-seed>.json
  sweeps/<model>/seed_<training-seed>/
    manifest.json
    cells/*.json
    preflight.json
    analysis.json
  summary.json
  summary.csv
  performance_context.json
  performance_context.csv
```

Each manifest records the protocol, experiment hash, Git revision and dirty
status, command, Python and package versions, SUMO backend/version, input-file
checksums, and checkpoint checksum. After acceptance, copy the complete bundle
to a new versioned directory under `results/`; do not edit frozen files in place.
