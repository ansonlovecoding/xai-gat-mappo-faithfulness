"""Run the complete v12 protocol, preserving per-stage timing and logs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dispatch_marl.provenance import atomic_write_json, runtime_provenance, sha256_file


def main():
    sumo_home = ROOT / "runs/runtime/sumo-1.20.0"
    os.environ["SUMO_HOME"] = str(sumo_home)
    os.environ["PATH"] = os.pathsep.join([str(sumo_home / "bin"),
                                        str(Path(sys.executable).parent), os.environ.get("PATH", "")])
    os.environ['DISPATCH_MARL_FORCE_TRACI'] = '1'
    os.environ['DISPATCH_MARL_DEVICE'] = 'cpu'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.chdir(ROOT)
    config = Path('configs/experiments/dissertation_v12_full_rerun.toml')
    run = Path('runs/dissertation_v12_sumo120')
    evidence = run / 'evidence'
    records = run / 'execution_records'
    records.mkdir(parents=True, exist_ok=True)
    record_path = records / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    record = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  config_sha256=sha256_file(config), provenance=runtime_provenance(ROOT),
                  device='cpu', threads=1, sumo_binary_sha256=sha256_file(sumo_home/'bin/sumo'),
                  sumo_source_revision=subprocess.check_output(
                      ['git', '-C', str(sumo_home), 'rev-parse', 'HEAD'], text=True).strip(),
                  stages=[])
    atomic_write_json(record_path, record)
    py = sys.executable
    stages = [('environment', [py, 'scripts/check_environment.py'])]
    for stage in ['train', 'select', 'evaluate', 'diagnose', 'sweep', 'robustness',
                  'preflight', 'analyze', 'summarize', 'controls',
                  'analyze-robustness', 'analyze-controls', 'audit']:
        stages.append((stage, [py, 'scripts/run_dissertation_experiments.py',
                       '--config', str(config), '--stage', stage, '--resume']))
    stages.extend([
        ('baselines', [py, 'scripts/run_matched_baselines.py', '--config', str(config),
                       '--output', str(evidence / 'matched_baselines.json')]),
        ('package', [py, 'scripts/package_dissertation_release.py', '--config', str(config)]),
        ('noop', [py, 'scripts/analyze_noop_absolute_def.py', '--evidence', str(evidence/'release'),
                  '--out', str(evidence/'supplementary_review')]),
    ])
    for name, command in stages:
        item = dict(stage=name, command=command, status='running',
                    started_utc=datetime.now(timezone.utc).isoformat())
        record['stages'].append(item)
        atomic_write_json(record_path, record)
        start = time.monotonic()
        print(f'STAGE {name}', flush=True)
        with (records / f'{record_path.stem}_{name}.log').open('w') as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        item.update(returncode=code, wall_seconds=time.monotonic()-start,
                    status='completed' if code == 0 else 'failed',
                    finished_utc=datetime.now(timezone.utc).isoformat())
        if code:
            record.update(status='failed', failed_stage=name)
            atomic_write_json(record_path, record)
            return code
        atomic_write_json(record_path, record)
    record.update(status='experiments_complete_document_update_pending',
                  finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_write_json(record_path, record)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
