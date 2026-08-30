# Preserved training results

Each subdirectory under `results/` is one "keeper" training run — a snapshot
of a `runs/mappo/<timestamp>/` directory that was interesting enough to save
under a stable, descriptive name. Contents mirror the format `train.py`
writes.

The point of `results/` vs. `runs/`:

- `runs/` is a scratch space — every `python scripts/train.py` call creates
  a fresh timestamped dir. Old ones can be safely deleted.
- `results/` is committed / archived. These are the runs the dissertation
  will cite.

Every keeper directory contains a `SUMMARY.md` explaining what the run was
for, which checkpoint is "the trained model", and how to reproduce the
reported numbers.

## Current keepers

| Directory | Scenario | Kind | Best pickups |
|---|---|---|---:|
| `mappo_central_park_reshaped_v1/` | central_park | single trained model | 7.67 ± 2.05 (ckpt_epoch_0050.pt) |
| `ab_centralised_critic_v1_seed42/` | central_park | A/B comparison | B (centralised, ckpt_50) beats A (decentralised) by +18 % mean pickups |
| `legacy_sumo120_seed42_v1/` | central_park | matched B1/B2 comparison + first H1–H4 sweep (SUMO 1.20, Intel mac) | B1 6.60 ± 2.24, B2 5.80 ± 1.72 (stochastic eval) |
| `story_freeze_v1/` | central_park | three-act story under the corrected freeze mechanism + max-AoI ladder (test demand, 8 seeds) | Act1 clean DEF −0.54; H5 mitigation rejected; decoupled head +0.11 (p=1e-4) |

Only the citable artefacts are committed (SUMMARY.md, configs, train logs,
eval JSONs, and each summary's cited checkpoint); periodic
`ckpt_epoch_*.pt` snapshots stay out of git — see `.gitignore`.

## Environment provenance (per keeper)

Numbers from different keepers are NOT cross-comparable — each row states
the SUMO version, machine, and observation layout that produced it:

| Keeper | SUMO | Machine | Obs layout | Loadable today? |
|---|---|---|---|---|
| `mappo_central_park_reshaped_v1/` | 1.27 | Apple Silicon | self 3-d / taxi 4-d | ✗ (obs layout predates AoI/velocity features) |
| `ab_centralised_critic_v1_seed42/` | 1.27 | Apple Silicon | self 5-d / taxi 4-d | ✗ (same reason) |
| `legacy_sumo120_seed42_v1/` | 1.20 | Intel mac | current | ✓ — but degradation = legacy NOISE mechanism, AOI_MAX_S=300 (WAMSN scale ×5 vs later) |
| `story_freeze_v1/` | 1.20 | Intel mac | current | ✓ — freeze mechanism, AOI_MAX_S=60 (definitive for the dissertation) |

> **Compatibility warning (July 2026):** both keepers were trained on
> older observation layouts (`reshaped_v1`: self 3-dim / taxi 4-dim;
> `ab_…_seed42`: self 5-dim / taxi 4-dim) and **cannot be loaded against
> the current env**, which uses self 5-dim / taxi 5-dim since neighbour
> AoI was added for WAMSN. Their reported numbers remain valid history
> (and were produced under SUMO 1.27 on the old Apple-Silicon machine),
> but any new comparison table must come from freshly trained
> checkpoints at the current code state.
