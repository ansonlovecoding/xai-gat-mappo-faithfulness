"""Repeat fixed-checkpoint analyses after four-decimal input rounding."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    analyses = sorted(args.root.glob('sweeps/*/*/analysis.json'))
    if not analyses:
        parser.error('no completed full-precision analyses found')

    def analyze(path: Path) -> None:
        output = path.with_name('analysis_rounded4.json')
        with path.with_name('precision_analysis.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT/'scripts/analyze_hypotheses.py'),
                            str(path.parent), '--input-round-decimals', '4',
                            '--output', str(output)], check=True, stdout=log,
                           stderr=subprocess.STDOUT)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(analyze, analyses))
    subprocess.run([sys.executable, str(ROOT/'scripts/summarize_precision_sensitivity.py'),
                    str(args.root)], check=True)


if __name__ == '__main__':
    main()
