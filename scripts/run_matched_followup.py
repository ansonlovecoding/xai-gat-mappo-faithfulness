"""Run the prospective matched-budget study with wall-clock and hardware records."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dispatch_marl.provenance import hardware_inventory, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stage', default='all', choices=['all', 'train', 'select',
                        'evaluate', 'framework'])
    args = parser.parse_args()
    config = ROOT / 'configs/experiments/dissertation_v11_matched_budget.toml'
    cfg = tomllib.loads(config.read_text())
    command = [sys.executable, 'scripts/run_dissertation_experiments.py',
               '--config', str(config), '--stage', args.stage, '--resume']
    if args.dry_run:
        return subprocess.call(command + ['--dry-run'], cwd=ROOT)
    started = datetime.now(timezone.utc)
    out = ROOT / cfg['output_root'] / 'execution_records'
    out.mkdir(parents=True, exist_ok=True)
    record = dict(started_utc=started.isoformat(), hardware=hardware_inventory(),
                  config_sha256=sha256_file(config), command=command, status='running')
    path = out / (started.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    path.write_text(json.dumps(record, indent=2)+'\n')
    t0 = time.monotonic()
    code = 1
    try:
        code = subprocess.call(command, cwd=ROOT)
        return code
    finally:
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      wall_seconds=time.monotonic()-t0, returncode=code,
                      status='completed' if code == 0 else 'failed_or_interrupted')
        path.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
