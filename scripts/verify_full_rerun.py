"""Check complete rerun coverage and identities before thesis packaging."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.root
    read = lambda p: json.loads(p.read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    empty = hashlib.sha256(b'').hexdigest()
    revisions, hardware, selected, errors = set(), set(), {}, []
    def check(condition, message):
        if not condition:
            errors.append(message)
    for model in ('B1_mlp', 'B2_gat', 'H5_gat_degraded'):
        for seed in range(42, 47):
            directory = run/'training'/model/f'seed_{seed}'
            manifest = read(directory/'manifest.json')
            check(read(directory/'training_stability.json')['ok'], f'stability {model}/{seed}')
            revisions.add(manifest['code_revision'])
            hardware.add(json.dumps(manifest['provenance']['hardware'], sort_keys=True))
            check(manifest['tracked_diff_sha256'] == empty, f'dirty training {model}/{seed}')
            logs = [json.loads(s) for s in (directory/'train_log.jsonl').read_text().splitlines() if s]
            check([x['epoch'] for x in logs] == list(range(50)), f'epoch coverage {model}/{seed}')
            checkpoint_hash = digest(directory/'ckpt_selected.pt')
            selected[model, seed] = checkpoint_hash
            for item in manifest['input_inventory']:
                check(digest(ROOT/item['path']) == item['sha256'], f'changed input {item["path"]}')
            evaluations = list((run/'evaluations'/model/f'seed_{seed}').glob('*.json'))
            check(len(evaluations) == 8, f'clean coverage {model}/{seed}')
            for path in evaluations:
                d = read(path)
                hardware.add(json.dumps(d['provenance']['hardware'], sort_keys=True))
                check(d['checkpoint_sha256'] == checkpoint_hash and len(d['per_episode']) == 6,
                      f'clean identity/episodes {path}')
    counts = {}
    for category, pattern, expected, episodes in (
        ('main', 'sweeps/*/*/cells/*.json', 400, 6),
        ('random_loss', 'robustness/*/*/*/cells/*.json', 240, 3),
        ('controls', 'faithfulness_controls/*/cells/*.json', 120, 3)):
        paths = list(run.glob(pattern))
        counts[category] = len(paths)
        check(len(paths) == expected, f'{category} coverage')
        for path in paths:
            d = read(path)
            check(len(d['per_episode']) == episodes, f'episode coverage {path}')
            if category == 'controls':
                check(d['checkpoint_sha256'] == selected[d['model_id'], d['training_seed']], f'control identity {path}')
    for path in run.glob('sweeps/*/*/manifest.json'):
        d = read(path); model = path.parent.parent.name; seed = int(path.parent.name[5:])
        check(d['checkpoint_sha256'] == selected[model, seed], f'sweep checkpoint identity {path}')
        check(d['tracked_diff_sha256'] == empty, f'dirty sweep {path}')
        revisions.add(d['code_revision'])
        check(read(path.with_name('preflight.json'))['ok'], f'preflight {path}')
    for path in run.glob('robustness/*/*/*/manifest.json'):
        d = read(path);model=path.parent.parent.name;seed=int(path.parent.name[5:])
        check(d['checkpoint_sha256'] == selected[model, seed], f'robustness identity {path}')
        check(d['tracked_diff_sha256'] == empty, f'dirty robustness {path}')
        revisions.add(d['code_revision'])
    check(revisions == {'2c604e73fe1501a9a1a370520857e12cdaeb74a8'}, 'source revision')
    check(len(hardware) == 1, 'hardware identity')
    diagnostics = list(run.glob('deterministic_diagnostics/*/*.json'))
    check(len(diagnostics) == 10, 'diagnostic coverage')
    check(all(x['completed_passenger_journeys'] > 0 for p in diagnostics for x in read(p)['per_episode']), 'diagnostic zero journeys')
    check(len(list(run.glob('sweeps/*/*/analysis_rounded4.json'))) == 10, 'precision coverage')
    baseline = read(run/'evidence/matched_baselines.json')
    check(len(baseline['records']) == 96, 'baseline coverage')
    payload = dict(ok=not errors, errors=errors, training_runs=len(selected), epochs_per_run=50,
                   clean_cells=120, diagnostic_cells=len(diagnostics), cells=counts,
                   source_revisions=sorted(revisions), hardware=[json.loads(x) for x in hardware])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
