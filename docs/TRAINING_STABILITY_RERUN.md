# Training stability correction and rerun protocol

## Status

The dissertation currently reports the archived `dissertation_v4` experiment.
Those results must remain unchanged until the complete replacement experiment
has finished. The corrected training protocol is defined in:

```text
configs/experiments/dissertation_v7.toml
```

Its outputs use `runs/dissertation_v7/`, so they cannot overwrite v4. The v5
and v6 outputs are development diagnostics and are not thesis results. V5
showed that the GAT schedule was too conservative for MLP. V6 showed that MLP
performance peaked near epoch 40 and drifted after epoch 50.

## Why the correction was needed

The clean GAT runs for seeds 42 and 44 learned useful policies early but then
collapsed. Their final ten training epochs averaged zero pickups, with longest
zero-pickup runs of 117 and 128 epochs. Inspection found five linked problems:

1. During rollout, the centralised critic pooled agents from one simulation
   step. During PPO, shuffled minibatches pooled agents from unrelated steps.
2. Large critic losses could dominate the shared encoder because the value
   update was not clipped.
3. Successful dispatch credit was included only in a team reward, so a taxi
   choosing no-op could receive the same immediate dispatch signal as the taxi
   whose action succeeded.
4. Constant learning rate and weak decision-state entropy regularisation let a
   useful policy continue drifting toward the no-op attractor.
5. Critic gradients entered the shared policy encoder at full strength, so
   value fitting could disturb representations already useful to the actor.

## Corrected training design

The v7 protocol applies the following controls:

1. Every agent step stores its environment-transition id. PPO minibatches keep
   complete transitions together, and the centralised critic pools only within
   each transition.
2. The critic uses PPO-style clipped value loss, while target KL can stop an
   update before it becomes too large.
3. Reported team reward is unchanged. The taxi whose dispatch succeeds receives
   an additional training-only difference credit.
4. Entropy is optimised over states that contain at least one valid request;
   forced no-op states do not dilute the exploration signal.
5. Only 10% of the critic gradient enters the shared encoder; the critic head
   still receives its full gradient and its forward value is unchanged.
6. Entropy regularisation increases only when decision entropy falls below
   0.20, with a coefficient cap of 0.20.
7. Learning rate decays linearly over 40 epochs. Validation uses eight fixed
   episodes rather than three.
8. A stability gate runs after checkpoint selection and rejects a seed before
   held-out evaluation if late-run capability has collapsed.

## Fixed v7 settings

| Setting | Value |
|---|---:|
| Training epochs | 40 |
| Initial learning rate | 0.0001 |
| Learning-rate schedule | linear decay |
| Entropy coefficient | 0.05 |
| Decision-entropy floor | 0.20 |
| Maximum entropy coefficient | 0.20 |
| Value-loss coefficient | 0.25 |
| Value clip | 0.20 |
| Target KL | 0.015 |
| Individual dispatch credit | 1.0 |
| Critic-to-encoder gradient scale | 0.10 |
| Validation episodes | 8 |

The table above is the GAT schedule. The MLP baseline uses 50 epochs, learning
rate 0.0003, and no learning-rate decay. Validation-only diagnostics showed
that the lower GAT learning rate left MLP near its initial policy, while 60
epochs reduced seed 43's final retention to 79.66%. The shorter budget keeps
the fixed 80% gate unchanged and avoids the observed late drift.

## Stability probes

The corrected clean GAT protocol was tested first on the two v4 seeds that had
the earliest and longest collapse. These are development diagnostics, not
dissertation results.

| Seed | v4 final-10 mean pickups | corrected probe final-10 | Longest zero run | Selected validation pickups | Final validation pickups | Retention |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.0 | 13.3 | 2 | 14.12 | 14.12 | 100.0% |
| 44 | 0.0 | 9.9 | 1 | 13.50 | 12.50 | 92.6% |

Both probes pass the fixed gate: final-ten mean pickups at least 5, no more
than four consecutive zero-pickup epochs, selected validation mean pickups at
least 5, and final validation retention of at least 80%. The 40-epoch budget
was chosen from validation diagnostics only; held-out test and faithfulness
results were not inspected. These probes show a substantial correction, but
only the complete three-model, three-seed rerun can replace the thesis results.

## Run the corrected experiment

Inspect commands first:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v7.toml --dry-run
```

Then train and select checkpoints:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v7.toml --stage train

.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v7.toml --stage select
```

The selection stage runs `check_training_stability.py` for every seed. Do not
start held-out evaluation if any seed fails. After all nine training runs pass,
continue with evaluation, faithfulness sweeps, preflight, analysis and thesis
regeneration.
