"""Test hypotheses H1–H4 on a severity sweep produced by sweep_severity.py.

Hypotheses (proposal §7.5, operationalised in §7.4):

  * **H1** — degradation severity ↑ ⇒ DEF ↓
             (one-sided Spearman over per-decision DEF vs cell severity)
  * **H2** — DEF declines *faster* than task performance (decoupling):
             faithfulness-degradation rate > performance-degradation rate,
             both measured relative to the clean cell of the same seed.
             (paired sign-flip permutation test over (level × seed) cells)
  * **H3** — degradation severity ↑ ⇒ WAMSN ↑
             (one-sided Spearman, per-decision, WAMSN vs severity)
  * **H4** — WAMSN and DEF negatively correlated
             (one-sided Spearman across pooled per-decision records)

All statistics are numpy-only (no scipy): Spearman = Pearson on ranks;
p-values by label permutation; CIs by nonparametric bootstrap over
decisions. Every test is reported per severity axis (`dropout_rate` and
`tunnel_noise`) because the two axes measure different things — H1/H3 on
the dropout axis vary *how often* telemetry fails, on the tunnel axis
*how wrong* it is when it fails.

Usage:
  python scripts/analyze_hypotheses.py runs/sweeps/<sweep-dir>
  python scripts/analyze_hypotheses.py <sweep-dir> --n-permutations 20000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------- statistics


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties get the mean of their rank range)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    if denom == 0:
        return 0.0
    return float((rx * ry).sum() / denom)


def spearman_permutation_p(
    x: np.ndarray, y: np.ndarray, alternative: str,
    n_permutations: int, rng: np.random.Generator,
) -> tuple[float, float]:
    """(rho, one-sided permutation p) for H0: no monotone association."""
    rho = spearman(x, y)
    y_perm = y.copy()
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(y_perm)
        r = spearman(x, y_perm)
        if alternative == "less" and r <= rho:
            count += 1
        elif alternative == "greater" and r >= rho:
            count += 1
    p = (count + 1) / (n_permutations + 1)
    return rho, float(p)


def signflip_p(
    deltas: np.ndarray, n_permutations: int, rng: np.random.Generator,
) -> float:
    """One-sided paired sign-flip test for H0: mean(delta) <= 0."""
    obs = deltas.mean()
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=len(deltas))
        if (deltas * signs).mean() >= obs:
            count += 1
    return float((count + 1) / (n_permutations + 1))


def bootstrap_ci(
    values: np.ndarray, n_boot: int, rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    means = np.array([
        values[rng.integers(0, len(values), len(values))].mean()
        for _ in range(n_boot)
    ])
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


# --------------------------------------------------------------- data loading


def load_sweep(sweep_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((sweep_dir / "manifest.json").read_text())
    cells = []
    for p in sorted(sweep_dir.glob("*.json")):
        if p.name in ("manifest.json", "analysis.json"):
            continue
        cells.append(json.loads(p.read_text()))
    if not cells:
        raise SystemExit(f"no cell JSONs found in {sweep_dir}")
    return manifest, cells


def decisions_frame(cells: list[dict]) -> dict[str, np.ndarray]:
    """Flatten per-decision records across cells into parallel arrays."""
    axis, level, seed = [], [], []
    def_, wamsn, drift, valid_res = [], [], [], []
    for c in cells:
        meta = c["cell"]
        for r in c.get("faith_records", []):
            axis.append(meta["axis"])
            level.append(meta["level"])
            seed.append(meta["seed"])
            def_.append(r["def"])
            wamsn.append(r["wamsn"])
            drift.append(r.get("drift", np.nan))
            valid_res.append(r["valid_reservations"])
    return {
        "axis": np.array(axis),
        "level": np.array(level, dtype=np.float64),
        "seed": np.array(seed),
        "def": np.array(def_, dtype=np.float64),
        "wamsn": np.array(wamsn, dtype=np.float64),
        "drift": np.array(drift, dtype=np.float64),
        "valid_res": np.array(valid_res, dtype=np.int64),
    }


# --------------------------------------------------------------- analyses


def _axis_mask(frame: dict, axis: str, include_clean: bool = True) -> np.ndarray:
    m = frame["axis"] == axis
    if include_clean:
        m = m | (frame["axis"] == "clean")
    return m


def hypothesis_h1_h3(
    frame: dict, axis: str, metric: str, alternative: str,
    n_permutations: int, rng: np.random.Generator,
) -> dict:
    """Shared machinery: monotone association between severity and a metric."""
    m = _axis_mask(frame, axis)
    if metric == "def":
        m = m & (frame["valid_res"] > 0)  # DEF undefined in no-op-only regime
    x = frame["level"][m]
    y = frame[metric][m]
    if len(x) < 10 or len(np.unique(x)) < 2:
        return {"n": int(len(x)), "note": "insufficient data"}
    rho, p = spearman_permutation_p(x, y, alternative, n_permutations, rng)
    return {"n": int(len(x)), "rho": rho, "p_one_sided": p,
            "per_level_mean": {
                f"{lv:g}": float(y[x == lv].mean()) for lv in np.unique(x)
            }}


def hypothesis_h2(
    cells: list[dict], axis: str,
    n_permutations: int, rng: np.random.Generator,
) -> dict:
    """Decoupling: faith-degradation rate minus perf-degradation rate > 0.

    Rates are computed per (severity level, seed) against the clean cell of
    the same seed:
        perf_rate  = (pickups_clean − pickups_s) / max(pickups_clean, ε)
        faith_rate = (DEF_clean − DEF_s) / max(|DEF_clean|, ε)
    """
    eps = 1e-9
    clean_by_seed: dict = {}
    for c in cells:
        if c["cell"]["axis"] == "clean":
            clean_by_seed[c["cell"]["seed"]] = c

    deltas, rows = [], []
    n_perf_floor = 0
    for c in cells:
        meta = c["cell"]
        if meta["axis"] != axis:
            continue
        clean = clean_by_seed.get(meta["seed"])
        if clean is None:
            continue
        p_clean = clean["mean_pickups"]
        d_clean = clean["faithfulness"].get("def_mean")
        p_s = c["mean_pickups"]
        d_s = c["faithfulness"].get("def_mean")
        if d_clean is None or d_s is None:
            continue
        if p_clean <= 0:
            # Clean performance is already zero (e.g. argmax on an entropy-
            # collapsed policy) — a performance-degradation *rate* is
            # undefined, and including the cell would make H2 vacuous.
            n_perf_floor += 1
            continue
        perf_rate = (p_clean - p_s) / max(p_clean, eps)
        faith_rate = (d_clean - d_s) / max(abs(d_clean), eps)
        deltas.append(faith_rate - perf_rate)
        rows.append({"level": meta["level"], "seed": meta["seed"],
                     "perf_rate": round(perf_rate, 4),
                     "faith_rate": round(faith_rate, 4)})
    if len(deltas) < 3:
        note = "insufficient data"
        if n_perf_floor > 0:
            note += (f" ({n_perf_floor} cells skipped: clean pickups = 0, "
                     "perf-rate undefined — run the sweep stochastically)")
        return {"n": len(deltas), "note": note}
    deltas_arr = np.array(deltas)
    return {
        "n": len(deltas),
        "n_skipped_perf_floor": n_perf_floor,
        "mean_delta_faith_minus_perf": float(deltas_arr.mean()),
        "p_one_sided": signflip_p(deltas_arr, n_permutations, rng),
        "cells": rows,
    }


def hypothesis_h4(
    frame: dict, n_permutations: int, rng: np.random.Generator,
) -> dict:
    """WAMSN vs DEF, pooled over all degraded cells, per-decision."""
    m = (frame["axis"] != "clean") & (frame["valid_res"] > 0)
    x = frame["wamsn"][m]
    y = frame["def"][m]
    if len(x) < 10:
        return {"n": int(len(x)), "note": "insufficient data"}
    if np.allclose(x, x[0]):
        return {"n": int(len(x)),
                "note": "WAMSN constant across decisions (e.g. AoI-unaware "
                        "checkpoint or no stale nodes attended) — H4 untestable"}
    rho, p = spearman_permutation_p(x, y, "less", n_permutations, rng)
    return {"n": int(len(x)), "rho": rho, "p_one_sided": p}


# --------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--stat-seed", type=int, default=0,
                        help="RNG seed for permutations/bootstrap only")
    args = parser.parse_args()

    manifest, cells = load_sweep(args.sweep_dir)
    frame = decisions_frame(cells)
    rng = np.random.default_rng(args.stat_seed)

    print(f"sweep:   {args.sweep_dir}")
    print(f"ckpt:    {manifest['checkpoint']}  (epoch {manifest.get('epoch')})")
    print(f"cells:   {len(cells)}  |  decisions with records: {len(frame['def'])}")
    print()

    # ---- descriptive table, per axis × level (pooled over seeds)
    print(f"{'axis':<14} {'level':>7} {'deg_rate':>9} {'pickups':>8} "
          f"{'DEF':>7} {'DEF 95% CI':>18} {'WAMSN':>7} {'drift':>7}")
    print("-" * 84)
    axes_levels = sorted({(c["cell"]["axis"], c["cell"]["level"]) for c in cells})
    for ax, lv in axes_levels:
        group = [c for c in cells
                 if c["cell"]["axis"] == ax and c["cell"]["level"] == lv]
        deg = float(np.mean([c["empirical_degradation_rate"] for c in group]))
        pk = float(np.mean([c["mean_pickups"] for c in group]))
        m = (frame["axis"] == ax) & (frame["level"] == lv) & (frame["valid_res"] > 0)
        defs = frame["def"][m]
        lo, hi = bootstrap_ci(defs, args.n_bootstrap, rng)
        m_all = (frame["axis"] == ax) & (frame["level"] == lv)
        wam = float(frame["wamsn"][m_all].mean()) if m_all.any() else float("nan")
        dr = frame["drift"][m_all]
        dr = float(np.nanmean(dr)) if m_all.any() else float("nan")
        print(f"{ax:<14} {lv:>7g} {deg:>9.4f} {pk:>8.2f} "
              f"{defs.mean() if len(defs) else float('nan'):>+7.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]   {wam:>7.3f} {dr:>7.4f}")
    print()

    # ---- hypothesis tests
    results: dict = {"manifest": {"checkpoint": manifest["checkpoint"],
                                  "git_rev": manifest.get("git_rev")}}
    for axis in ("dropout_rate", "tunnel_noise"):
        results[f"H1_{axis}"] = hypothesis_h1_h3(
            frame, axis, "def", "less", args.n_permutations, rng)
        results[f"H2_{axis}"] = hypothesis_h2(
            cells, axis, args.n_permutations, rng)
        results[f"H3_{axis}"] = hypothesis_h1_h3(
            frame, axis, "wamsn", "greater", args.n_permutations, rng)
    results["H4_pooled"] = hypothesis_h4(frame, args.n_permutations, rng)

    def _verdict(r: dict, label: str) -> str:
        if "note" in r:
            return f"{label}: n/a ({r['note']}, n={r['n']})"
        p = r.get("p_one_sided")
        stat = r.get("rho", r.get("mean_delta_faith_minus_perf"))
        sig = "SUPPORTED" if p is not None and p < 0.05 else "not supported"
        return f"{label}: {sig}  (stat={stat:+.3f}, p={p:.4f}, n={r['n']})"

    print("hypothesis tests (α = 0.05, one-sided):")
    for axis in ("dropout_rate", "tunnel_noise"):
        print(" ", _verdict(results[f"H1_{axis}"], f"H1 (DEF ↓ with {axis})"))
        print(" ", _verdict(results[f"H2_{axis}"], f"H2 (faith declines faster, {axis})"))
        print(" ", _verdict(results[f"H3_{axis}"], f"H3 (WAMSN ↑ with {axis})"))
    print(" ", _verdict(results["H4_pooled"], "H4 (WAMSN–DEF negative, pooled)"))

    out_path = args.sweep_dir / "analysis.json"
    out_path.write_text(json.dumps(results, indent=2))
    print()
    print(f"analysis: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
