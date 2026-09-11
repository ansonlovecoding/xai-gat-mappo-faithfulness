"""Supplementary no-op analysis from immutable, archived decision records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/dissertation_v10_corrected/release"
OUT = ROOT / "results/dissertation_v10_corrected/supplementary_review"


def main():
    global EVIDENCE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=EVIDENCE)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    EVIDENCE, OUT = args.evidence.resolve(), args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sources, tests = [], [], []
    rng = np.random.default_rng(20260912)
    for archive in sorted((EVIDENCE / "audit_records").glob("*_cells.tar.gz")):
        checkpoint = archive.name.removesuffix("_cells.tar.gz")
        sources.append({"path": str(archive.relative_to(ROOT)),
                        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
        with tarfile.open(archive) as tar:
            cells = [json.load(tar.extractfile(m)) for m in tar.getmembers()
                     if m.isfile() and m.name.endswith('.json')
                     and (Path(m.name).name.startswith('clean_')
                          or Path(m.name).name.startswith('outage_duration_60_'))]
        # Check archived condition metadata, independent of member naming.
        conditions = {}
        for cell in cells:
            level = float(cell['outage_duration_s'])
            if level not in (0, 60):
                continue
            conditions.setdefault(level, []).append(cell)
        if set(conditions) != {0, 60}:
            raise ValueError(f'Missing clean/60s condition: {checkpoint}, {conditions.keys()}')
        for level, selected in sorted(conditions.items()):
            for metric in ('primary_def', 'primary_def_m'):
                blocks, count, candidates = [], 0, 0
                for cell in selected:
                    episodes = {}
                    for record in cell['faith_records']:
                        if record.get('action') != 0 or record.get('valid_reservations', 0) <= 0:
                            continue
                        candidates += 1
                        value = record.get(metric)
                        if value is None or not np.isfinite(value):
                            continue
                        episodes.setdefault(record['episode'], []).append(float(value))
                        count += 1
                    blocks.extend(np.mean(values) for values in episodes.values())
                values = np.asarray(blocks)
                if not len(values):
                    raise ValueError(f'No eligible records for {checkpoint}/{level}/{metric}')
                boot = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
                lo, hi = np.quantile(boot, [.025, .975])
                rows.append(dict(checkpoint=checkpoint, condition='clean' if level == 0 else 'outage_60s',
                                 metric=metric, mean=float(values.mean()), ci95_low=float(lo),
                                 ci95_high=float(hi), n_episode_blocks=len(values),
                                 n_decisions=count, n_missing_metric=candidates-count))
        analysis = json.loads((EVIDENCE / 'sweep_evidence' / checkpoint / 'analysis.json').read_text())
        robust = analysis['robust']
        for hypothesis in ('H1', 'H2', 'H3', 'H4'):
            detail = (robust['H1_robust'] if hypothesis == 'H1' else
                      robust['H3_robust'] if hypothesis == 'H3' else
                      robust['H4_within_episode'] if hypothesis == 'H4' else
                      analysis['H2_outage_duration'])
            tests.append(dict(checkpoint=checkpoint, hypothesis=hypothesis,
                              statistic=detail.get('rho', detail.get('mean_rho_within_episode', detail.get('mean_delta_faith_minus_perf'))),
                              raw_p=robust['confirmatory_family_raw_p'][hypothesis],
                              holm_p=robust['confirmatory_family_holm_p'][hypothesis],
                              n=detail.get('n', detail.get('n_episodes')),
                              n_episode_blocks=detail.get('n_episode_blocks', detail.get('n_episodes')),
                              ci95=json.dumps(detail.get('rho_cluster_ci95'))))
    if not rows:
        raise ValueError('No archived clean/60-second cells found')
    for name, data in [('noop_absolute_def', rows), ('hypothesis_statistics', tests)]:
        with (OUT / f'{name}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    (OUT / 'analysis_manifest.json').write_text(json.dumps(dict(
        analysis='Exploratory, post-review no-op absolute DEF; no new training',
        sources=sources, bootstrap_seed=20260912, bootstrap_draws=10000,
        sampling='Existing cadence plus stale-exposure schedule; not all policy decisions',
        unit='Equal-weight episode means within each fixed checkpoint and condition',
        intervals='Pointwise percentile intervals, not simultaneous or population intervals',
        exclusions='No-op with no available request; missing or non-finite metric',
        rows=rows), indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
