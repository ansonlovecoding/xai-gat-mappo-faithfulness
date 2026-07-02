# Training on Google Colab

Move training to a Colab GPU (usually a T4 or L4), then bring the trained
`.pt` checkpoint back to your local machine for evaluation and visual
inspection.

## Why bother

- **Free GPU time**. Colab's T4 is ~5–10× faster than the M-series MPS for a
  torch model this size. A 500-epoch run drops from ~15 min to ~2 min.
- **Long headless runs** without tying up your laptop.
- **Reproducibility**: the notebook doubles as a record of the exact commands
  that produced a given checkpoint.

## Prerequisites

- The project pushed to a Git host (GitHub / GitLab / Bitbucket). Colab's
  fastest bootstrap path is `git clone`. If your repo is private, use a
  fine-grained deploy token, or fall back to Drive upload (below).
- Committed scenario files (`scenarios/yubei/*/*.osm.xml` + `.net.xml` +
  `.rou.xml` + `.sumocfg` + `tunnels.json` + `_taxis.rou.xml`). These are
  already committed in this project.

## Notebook cells

Paste each fenced block into a fresh Colab cell in order.

### 1. Runtime setup

Set the runtime type to **GPU (T4)** via *Runtime → Change runtime type* before
running any cells.

### 2. Install SUMO 1.27.0 (matches our local version)

```bash
%%bash
set -eux
# Latest SUMO from the DLR PPA, which tracks the 1.27.x line at time of writing.
add-apt-repository -y ppa:sumo/stable
apt-get update -qq
apt-get install -y sumo sumo-tools sumo-doc
# Sanity-check the version.
sumo --version | head -3
```

If the PPA install fails (occasionally the mirror has network issues), fall
back to the older Ubuntu-packaged version:

```bash
%%bash
apt-get install -y sumo sumo-tools
sumo --version | head -3
```

An older SUMO usually still works, but you must pin `sumolib` and `traci` in
your requirements.txt to match — otherwise TraCI will complain about protocol
version mismatches.

### 3. Set `SUMO_HOME`

```python
import os
# Debian/Ubuntu installs land SUMO here.
os.environ["SUMO_HOME"] = "/usr/share/sumo"
os.environ["PATH"] = f"{os.environ['SUMO_HOME']}/bin:{os.environ['PATH']}"
!echo $SUMO_HOME && ls $SUMO_HOME/tools | head -3
```

Our `src/dispatch_marl/_sumo.py` already checks `/usr/share/sumo` as one of
its fallback candidates, so this step is belt-and-braces.

### 4. Clone the repo and install Python deps

```bash
%%bash
set -eux
cd /content
git clone https://github.com/<your-user>/<your-repo>.git dispatch
cd dispatch
# Do NOT reinstall torch — Colab's pre-installed build includes CUDA. A plain
# `pip install torch` would replace it with a CPU wheel and silently kill
# GPU acceleration. Install everything else, then verify CUDA.
pip install -q -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'CUDA missing'; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
```

Replace the URL with your real one. If the repo is private, use
`git clone https://<TOKEN>@github.com/...` with a repo-scoped token.

**If you don't have a Git host**: mount Drive, upload a zip of the project,
and unzip:

```python
from google.colab import drive
drive.mount("/content/drive")
!cp /content/drive/MyDrive/dispatch.zip /content/
!cd /content && unzip -q dispatch.zip
```

### 5. Sanity-check the env

```bash
%%bash
cd /content/dispatch
python scripts/smoke_test.py 2>&1 | tail -5
```

Should print `OK — SUMO end-to-end pipeline is healthy.`

### 6. Train

```bash
%%bash
cd /content/dispatch
python scripts/train.py \
  --area central_park \
  --epochs 500 \
  --save-every 50 \
  --pickup-reward 10.0 \
  --dispatch-reward 0.5 \
  --wait-lambda 0.001 \
  --device cuda
```

Training writes to `runs/mappo/central_park_<ts>/`, same layout as local.

### 7. Zip the run directory and download it

```python
import shutil, glob
latest = sorted(glob.glob("/content/dispatch/runs/mappo/*"))[-1]
out = shutil.make_archive("/content/latest_run", "zip", latest)
print("archive:", out)
from google.colab import files
files.download(out)
```

Or save it to Drive to avoid the browser download:

```python
!cp /content/latest_run.zip /content/drive/MyDrive/
```

### 8. Locally: unpack and evaluate

Back on your Mac:

```bash
cd "path/to/project"
mkdir -p runs/mappo/from_colab
unzip -o ~/Downloads/latest_run.zip -d runs/mappo/from_colab
python scripts/eval_policy.py \
  runs/mappo/from_colab/ckpt_epoch_0499.pt \
  --episodes 5
```

The eval script already overrides `PolicyConfig.device` to the local
device (MPS on Apple Silicon), so a CUDA-trained checkpoint loads cleanly.
Verify with the summary line at the end, which reports the actual runtime
device.

## Common gotchas

- **Colab session timeout**: free tier disconnects after ~90 min of
  inactivity and hard-caps at 12 h. Long runs (thousands of epochs) should
  checkpoint every 50 epochs (which our `--save-every 50` does), so you can
  resume manually if needed.
- **SUMO version mismatch**: if you install a different SUMO in Colab than
  we have locally, `sumolib` / `traci` on both sides must match. The safest
  approach is to bump `requirements.txt` to whatever the Colab SUMO reports
  and rerun `pip install` locally.
- **`weights_only=False` warning**: torch >= 2.4 warns on `torch.load`
  because we pickle non-tensor Python objects (the config dicts). Safe here
  since we control the checkpoints. Suppress by loading through
  `torch.load(path, weights_only=False)` explicitly (already done in
  `eval_policy.py`).
- **Determinism**: don't expect bit-identical numbers between CUDA and MPS —
  reduction ops differ. Aggregate metrics over multiple eval episodes
  (`--episodes 5+`) to reduce noise.
- **Randomness in the env**: SUMO's `randomTrips.py` + fixed seed makes the
  background traffic reproducible, but the trained policy under different
  torch RNG on CUDA vs MPS will still produce slightly different actions.
  Same caveat as above.
