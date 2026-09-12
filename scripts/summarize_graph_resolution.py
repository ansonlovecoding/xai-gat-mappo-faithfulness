"""Describe eligible graph sizes in scored main-sweep decisions."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for model_dir in sorted((args.root/'sweeps').iterdir()):
        records = [record for path in sorted(model_dir.glob('*/cells/*.json'))
                   for record in json.loads(path.read_text())['faith_records']
                   if record.get('primary_def') is not None and record.get('valid_reservations', 0) > 0]
        for field in ('valid_taxis', 'valid_reservations', 'valid_non_self_nodes'):
            values = np.array([r[field] for r in records])
            rows.append(dict(model=model_dir.name, field=field, n=len(values),
                             mean=float(values.mean()), median=float(np.median(values)),
                             p05=float(np.quantile(values, .05)), p95=float(np.quantile(values, .95))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        source='Main-sweep records with primary_def and at least one available request, all conditions; decision-weighted descriptive distribution',
        rows=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
