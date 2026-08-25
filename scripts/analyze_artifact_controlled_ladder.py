"""Artifact-controlled severity ladder + exposure-conditional statistics (P5).

Motivation
----------
`results/story_freeze_v1/audit/type_matched_control.json` establishes that
the clean-condition margin-DEF of -0.554 is 98 % a *composition artifact*:
uniform random occlusion sets hit reservation nodes -- and therefore delete
candidate actions, clamping the logit margin to -cap -- far more often than
the attention top-k does. Under a composition-matched random baseline the
clean score is -0.009.

That control was only ever run on the clean condition. The severity ladder
(`runs/sweeps/B2_aoi_ladder_v2`) is still scored under the uniform baseline,
so the ladder cannot be reported as-is: it would present the metric the
audit invalidated.

Recomputing the true type-matched baseline requires fresh counterfactual
policy forwards, i.e. a full sweep re-run (`sweep_severity.py
--random-baseline type_matched`). This script provides what *can* be
recovered from the already-recorded per-decision data, so the ladder has a
defensible artifact control either way:

  1. **Clamp-free subset.** Decisions where neither the top-k nor any random
     occlusion set clamped. The action-deletion mechanism is absent by
     construction. Exact, assumption-free, but small (n ~ 90).
  2. **Clamp-differential adjustment.** OLS of margin-DEF on the per-decision
     clamp differential (clamp_rand - clamp_topk) with level dummies, read
     off at differential = 0. Uses all ~17.7k decisions; an estimate, not a
     protocol, and reported as such.
  3. **Exposure-conditional statistics.** Stale-node exposure rate, and
     WAMSN / margin-DEF conditional on exposure, per level. These need no
     adjustment -- WAMSN never involves occlusion -- so they are exact and
     are the paper's primary degradation estimand.
  4. **H1 / H4 re-tests** on the adjusted and exposure-conditional data,
     with episode-clustered inference throughout.

Usage:
  python scripts/analyze_artifact_controlled_ladder.py runs/sweeps/B2_aoi_ladder_v2
  python scripts/analyze_artifact_controlled_ladder.py <sweep> --out results/.../audit/x.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_BOOT = 10000
N_PERM = 10000
# --------------------------------------------------------------- loading


def load_records(sweep_dir: Path) -> list[dict]:
    """Flatten every cell's per-decision records, tagging level/seed/cluster."""
    recs: list[dict] = []
    cell_dir = sweep_dir / "cells"
    candidates = cell_dir.glob("*.json") if cell_dir.is_dir() else sweep_dir.glob("*.json")
    for p in sorted(candidates):
        cell = json.loads(p.read_text())
        if not (isinstance(cell, dict) and "cell" in cell and "faith_records" in cell):
            continue
        c = cell["cell"]
        for r in cell.get("faith_records", []):
            r["_level"] = float(c["level"])
            r["_axis"] = c["axis"]
            r["_seed"] = int(c["seed"])
            # Episode is the resampling unit: decisions inside one episode
            # share demand, geography and policy state.
            r["_cluster"] = (float(c["level"]), int(c["seed"]), int(r["episode"]))
            recs.append(r)
    if not recs:
        raise SystemExit(f"no per-decision records found in {sweep_dir}")
    if "clamp_topk" not in recs[0]:
        raise SystemExit(
            f"{sweep_dir} is not P2-instrumented (no clamp_topk field) -- "
            "the artifact control needs clamp flags; use B2_aoi_ladder_v2"
        )
    return recs


# --------------------------------------------------------- cluster resampling


def cluster_boot_mean(
    values: np.ndarray, clusters: np.ndarray, rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> tuple[float, float, float]:
    """Mean of `values` with a cluster (episode) bootstrap 95 % CI.

    Vectorised: a resampled mean is sum(picked cluster sums) / sum(picked
    cluster counts), so the whole bootstrap is two matrix gathers.
    """
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    _, inv = np.unique(clusters, return_inverse=True)
    n_g = int(inv.max()) + 1
    if n_g < 2:
        return float(values.mean()), float("nan"), float("nan")
    sums = np.bincount(inv, weights=values, minlength=n_g)
    cnts = np.bincount(inv, minlength=n_g).astype(float)
    pick = rng.integers(0, n_g, size=(n_boot, n_g))
    boots = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return (float(values.mean()),
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


# ------------------------------------------------- clamp-differential model


def clamp_adjusted_levels(recs: list[dict], rng: np.random.Generator) -> dict:
    """Margin-DEF per level, adjusted to zero clamp differential.

    Design matrix: intercept-free level dummies + the clamp differential
    d = clamp_rand - clamp_topk. The coefficient on each dummy is that
    level's margin-DEF at d = 0, i.e. with the action-deletion asymmetry
    removed. Episode-clustered bootstrap for CIs.
    """
    levels = sorted({r["_level"] for r in recs})
    idx = {lv: i for i, lv in enumerate(levels)}

    y = np.array([r["def_m"] for r in recs], dtype=float)
    d = np.array([r["clamp_rand"] - r["clamp_topk"] for r in recs], dtype=float)
    X = np.zeros((len(recs), len(levels) + 1))
    for i, r in enumerate(recs):
        X[i, idx[r["_level"]]] = 1.0
    X[:, -1] = d
    clusters = np.array([hash(r["_cluster"]) for r in recs])

    beta = np.linalg.solve(X.T @ X, X.T @ y)

    # Cluster bootstrap via per-cluster Gram matrices: a resampled OLS fit is
    # solve(sum of picked X'X blocks, sum of picked X'y blocks). Avoids
    # rebuilding a 17k-row design matrix on every draw.
    _, inv = np.unique(clusters, return_inverse=True)
    n_g = int(inv.max()) + 1
    p = X.shape[1]
    gram = np.zeros((n_g, p, p))
    xty = np.zeros((n_g, p))
    for g in range(n_g):
        Xg = X[inv == g]
        yg = y[inv == g]
        gram[g] = Xg.T @ Xg
        xty[g] = Xg.T @ yg
    n_draws = N_BOOT // 10
    boots = np.empty((n_draws, p))
    for b in range(n_draws):
        pick = rng.integers(0, n_g, n_g)
        A = gram[pick].sum(axis=0)
        rhs = xty[pick].sum(axis=0)
        try:
            boots[b] = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            boots[b] = np.linalg.lstsq(A, rhs, rcond=None)[0]

    out = {
        "model": "def_m ~ level_dummies + (clamp_rand - clamp_topk), "
                 "read off at differential = 0",
        "clamp_differential_coef": float(beta[-1]),
        "clamp_differential_coef_ci95": [
            float(np.percentile(boots[:, -1], 2.5)),
            float(np.percentile(boots[:, -1], 97.5)),
        ],
        "n_decisions": int(len(recs)),
        "n_episode_clusters": int(n_g),
        "by_level": {},
    }
    for lv in levels:
        j = idx[lv]
        out["by_level"][f"{lv:g}"] = {
            "def_m_adjusted": float(beta[j]),
            "ci95": [float(np.percentile(boots[:, j], 2.5)),
                     float(np.percentile(boots[:, j], 97.5))],
        }
    return out


# ------------------------------------------------------------ hypothesis tests


def episode_block_trend(
    recs: list[dict], value_key: str, rng: np.random.Generator,
) -> dict:
    """Rank-vs-severity trend statistic with an episode-block permutation null.

    The statistic is the correlation between the *ranks* of the per-decision
    value and the numeric severity level -- a Spearman-type trend coefficient.
    Because the value ranks are fixed under the null, they are computed once
    and each permutation reduces to a single dot product, which keeps 10k
    block permutations over ~18k decisions cheap.

    The null shuffles level labels between whole episodes, never within, so
    the test respects the clustering induced by shared demand and geography.
    """
    vals = np.array([r[value_key] for r in recs], dtype=float)
    lvls = np.array([r["_level"] for r in recs], dtype=float)
    ok = np.isfinite(vals)
    vals, lvls = vals[ok], lvls[ok]
    keys = [r["_cluster"] for r, k in zip(recs, ok) if k]
    if vals.size < 10:
        return {"n": int(vals.size), "rho": float("nan"), "p_perm": float("nan")}

    # Average ranks so ties (e.g. the many WAMSN == 0 decisions) are handled.
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(vals.size, dtype=float)
    ranks[order] = np.arange(1, vals.size + 1, dtype=float)
    uv, first, counts = np.unique(vals, return_inverse=True, return_counts=True)
    tie_mean = np.bincount(first, weights=ranks) / counts
    ranks = tie_mean[first]
    rc = ranks - ranks.mean()
    rc_norm = np.sqrt((rc ** 2).sum())
    if rc_norm == 0:
        return {"n": int(vals.size), "rho": 0.0, "p_perm": 1.0,
                "note": "value is constant; no trend estimable"}

    _, inv = np.unique(np.array([hash(k) for k in keys]), return_inverse=True)
    n_b = int(inv.max()) + 1
    block_lvl = np.zeros(n_b)
    block_lvl[inv] = lvls  # one label per block by construction

    def _stat(level_vec: np.ndarray) -> float:
        lc = level_vec - level_vec.mean()
        den = rc_norm * np.sqrt((lc ** 2).sum())
        return float((rc * lc).sum() / den) if den > 0 else 0.0

    obs = _stat(lvls)
    hits = 0
    for _ in range(N_PERM):
        if abs(_stat(rng.permutation(block_lvl)[inv])) >= abs(obs):
            hits += 1
    return {
        "n": int(vals.size),
        "n_episode_blocks": n_b,
        "rho": obs,
        "p_perm": float((hits + 1) / (N_PERM + 1)),
        "p_perm_resolution": float(1.0 / (N_PERM + 1)),
    }


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks; ties get the mean of their rank range.

    Tie correction is not optional here: WAMSN is exactly 0 for ~90 % of
    decisions, so naive argsort-of-argsort ranks assign that tied block an
    arbitrary increasing sequence and can flip the sign of the correlation.
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < sx.size:
        j = i
        while j + 1 < sx.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _avg_ranks(a), _avg_ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def within_episode_association(
    recs: list[dict], value_key: str, min_n: int = 8,
) -> dict:
    """Mean within-episode Spearman(WAMSN, value) -- the H4 estimator.

    Matches `scripts/analyze_hypotheses.py::stratified_h4` (min 8 scored
    decisions per episode, episodes with constant WAMSN dropped) so the two
    pipelines are directly comparable, and adds an exact sign test on the
    episode-level rhos.
    """
    by_ep: dict = {}
    for r in recs:
        if r["valid_reservations"] > 0 and np.isfinite(r[value_key]):
            by_ep.setdefault(r["_cluster"], []).append(r)

    rhos = []
    for rs in by_ep.values():
        w = np.array([r["wamsn"] for r in rs], dtype=float)
        v = np.array([r[value_key] for r in rs], dtype=float)
        if w.size < min_n or np.allclose(w, w[0]):
            continue
        rho = _spearman(w, v)
        if np.isfinite(rho):
            rhos.append(rho)
    if len(rhos) < 5:
        return {"n_episodes": len(rhos), "note": "insufficient usable episodes"}
    a = np.array(rhos)
    n_neg = int((a < 0).sum())
    n = a.size
    from math import comb
    tail = sum(comb(n, k) for k in range(n_neg, n + 1)) / 2 ** n
    return {
        "n_episodes": n,
        "mean_rho": float(a.mean()),
        "median_rho": float(np.median(a)),
        "frac_negative": float(n_neg / n),
        "p_sign_test_two_sided": float(min(1.0, 2 * tail)),
    }


# ------------------------------------------------------------------- exposure


def exposure_stats(recs: list[dict], rng: np.random.Generator) -> dict:
    """Exact, occlusion-free degradation statistics per level.

    Two distinct exposure notions are reported because the paper conflated
    them: `stale_visible` (this decision's graph contains >= 1 stale vehicle
    node -- the quantity WAMSN is conditional on) and `stale_in_top3` (a
    stale node reached the explanation's top 3).
    """
    out: dict = {"by_level": {}}
    for lv in sorted({r["_level"] for r in recs}):
        sub = [r for r in recs if r["_level"] == lv]
        exposed = [r for r in sub if r.get("n_stale_veh", 0) > 0]
        cl = np.array([hash(r["_cluster"]) for r in sub])
        clx = np.array([hash(r["_cluster"]) for r in exposed])
        w_all = np.array([r["wamsn"] for r in sub], dtype=float)
        entry = {
            "n_decisions": len(sub),
            "stale_visible_rate": float(len(exposed) / len(sub)),
            "wamsn_pooled": dict(zip(
                ("mean", "ci_lo", "ci_hi"), cluster_boot_mean(w_all, cl, rng))),
        }
        if exposed:
            w_x = np.array([r["wamsn"] for r in exposed], dtype=float)
            d_x = np.array([r["def_m"] for r in exposed], dtype=float)
            entry["n_exposed"] = len(exposed)
            entry["wamsn_conditional"] = dict(zip(
                ("mean", "ci_lo", "ci_hi"), cluster_boot_mean(w_x, clx, rng)))
            entry["def_m_conditional_uniform"] = dict(zip(
                ("mean", "ci_lo", "ci_hi"), cluster_boot_mean(d_x, clx, rng)))
            entry["stale_in_top3_rate_conditional"] = float(
                np.mean([bool(r.get("stale_in_top3")) for r in exposed]))
            aoi = np.array([r.get("max_aoi_s", 0.0) for r in exposed], dtype=float)
            entry["max_aoi_s"] = {"p50": float(np.percentile(aoi, 50)),
                                  "p90": float(np.percentile(aoi, 90)),
                                  "max": float(aoi.max())}
        else:
            entry["n_exposed"] = 0
        out["by_level"][f"{lv:g}"] = entry
    return out


# ----------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: <sweep_dir>/typematched_ladder.json)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    recs = load_records(args.sweep_dir)
    scored = [r for r in recs if r["valid_reservations"] > 0]

    print(f"records: {len(recs)}  scored (>=1 valid reservation): {len(scored)}")
    print()

    out: dict = {
        "sweep_dir": str(args.sweep_dir),
        "n_records": len(recs),
        "n_scored": len(scored),
        "n_boot": N_BOOT,
        "n_perm": N_PERM,
    }

    # ---- 0. uniform baseline ladder, as currently reported
    print("0. UNIFORM-BASELINE LADDER (the artifact-prone metric, for reference)")
    out["uniform_ladder"] = {}
    for lv in sorted({r["_level"] for r in recs}):
        sub = [r for r in recs if r["_level"] == lv]
        v = np.array([r["def_m"] for r in sub], dtype=float)
        cl = np.array([hash(r["_cluster"]) for r in sub])
        m, lo, hi = cluster_boot_mean(v, cl, rng)
        out["uniform_ladder"][f"{lv:g}"] = {
            "def_m": m, "ci95": [lo, hi], "n": len(sub),
            "clamp_topk_mean": float(np.mean([r["clamp_topk"] for r in sub])),
            "clamp_rand_mean": float(np.mean([r["clamp_rand"] for r in sub])),
        }
        print(f"   level {lv:>4g}s: def_m {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"clamp topk {np.mean([r['clamp_topk'] for r in sub]):.3f} / "
              f"rand {np.mean([r['clamp_rand'] for r in sub]):.3f}")

    # ---- 1. clamp-free subset: exact, assumption-free, small
    print()
    print("1. CLAMP-FREE SUBSET (no action deletion on either side; exact)")
    free = [r for r in recs if r["clamp_topk"] == 0.0 and r["clamp_rand"] == 0.0]
    v = np.array([r["def_m"] for r in free], dtype=float)
    cl = np.array([hash(r["_cluster"]) for r in free])
    m, lo, hi = cluster_boot_mean(v, cl, rng)
    out["clamp_free_subset"] = {
        "n": len(free), "frac_of_all": float(len(free) / len(recs)),
        "def_m": m, "ci95": [lo, hi],
        "caveat": "small and selected (clamp-free decisions have fewer "
                  "competing reservations); corroborates the type-matched "
                  "result, does not replace it",
    }
    print(f"   n = {len(free)} ({len(free)/len(recs):.1%} of decisions)  "
          f"def_m {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # ---- 2. clamp-differential adjusted ladder
    print()
    print("2. CLAMP-DIFFERENTIAL ADJUSTED LADDER (all decisions; model-based)")
    adj = clamp_adjusted_levels(recs, rng)
    out["clamp_adjusted_ladder"] = adj
    print(f"   clamp-differential coef: {adj['clamp_differential_coef']:+.4f} "
          f"{[round(x, 4) for x in adj['clamp_differential_coef_ci95']]}")
    for lv, e in adj["by_level"].items():
        print(f"   level {lv:>4}s: def_m_adj {e['def_m_adjusted']:+.4f} "
              f"[{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]")

    # ---- 3. exposure-conditional statistics
    print()
    print("3. EXPOSURE-CONDITIONAL STATISTICS (exact; primary estimand)")
    exp = exposure_stats(recs, rng)
    out["exposure"] = exp
    for lv, e in exp["by_level"].items():
        line = (f"   level {lv:>4}s: stale-visible {e['stale_visible_rate']:.3f}  "
                f"WAMSN pooled {e['wamsn_pooled']['mean']:.4f}")
        if e["n_exposed"]:
            line += (f"  | exposed {e['wamsn_conditional']['mean']:.4f} "
                     f"(n={e['n_exposed']})  "
                     f"stale-in-top3 {e['stale_in_top3_rate_conditional']:.1%}")
        print(line)

    # ---- 4. H1 / H3 / H4 re-tests
    print()
    print("4. HYPOTHESIS RE-TESTS (episode-clustered)")
    degraded = [r for r in recs if r["_axis"] != "clean"]
    exposed_only = [r for r in recs if r.get("n_stale_veh", 0) > 0]

    # H4 is reported for BOTH DEF variants: the paper declares margin-DEF
    # primary but quotes the probability-DEF effect size, and the two differ
    # by ~3x. Both are computed here so the paper can be made consistent.
    clamp_free_deg = [r for r in degraded
                      if r["clamp_topk"] == 0.0 and r["clamp_rand"] == 0.0]
    tests = {
        "H1_uniform_def_m_vs_level_all": episode_block_trend(recs, "def_m", rng),
        "H1_uniform_def_m_vs_level_exposed_only":
            episode_block_trend(exposed_only, "def_m", rng),
        "H3_wamsn_vs_level_all": episode_block_trend(recs, "wamsn", rng),
        "H3_wamsn_vs_level_degraded_only":
            episode_block_trend(degraded, "wamsn", rng),
        "H4_within_episode_prob_def_degraded":
            within_episode_association(degraded, "def"),
        "H4_within_episode_margin_def_degraded":
            within_episode_association(degraded, "def_m"),
        "H4_within_episode_margin_def_clamp_free":
            within_episode_association(clamp_free_deg, "def_m"),
    }
    out["tests"] = tests
    for name, t in tests.items():
        if "rho" in t:
            print(f"   {name}: n={t['n']} rho={t['rho']:+.4f} "
                  f"p={t['p_perm']:.5f} (floor {t['p_perm_resolution']:.5f})")
        else:
            print(f"   {name}: episodes={t['n_episodes']} "
                  f"mean_rho={t.get('mean_rho', float('nan')):+.4f} "
                  f"frac_neg={t.get('frac_negative', float('nan')):.3f} "
                  f"p_sign={t.get('p_sign_test_two_sided', float('nan')):.5f}")

    # ---- 5. severity-manipulation check
    print()
    print("5. SEVERITY-MANIPULATION CHECK (did the ladder actually vary AoI?)")
    exposed = [r for r in recs if r.get("n_stale_veh", 0) > 0]
    aoi_all = np.array([r["max_aoi_s"] for r in exposed], dtype=float)
    cap = 60.0  # env.AOI_MAX_S
    sat = float(np.mean(aoi_all >= cap - 1e-6)) if aoi_all.size else float("nan")
    manip = {
        "aoi_censoring_cap_s": cap,
        "note": "max_aoi_s is reconstructed from the observation feature "
                "aoi_norm = clip(AoI / AOI_MAX_S, 0, 1), so it is RIGHT-CENSORED "
                "at AOI_MAX_S. WAMSN's staleness weight clip(AoI/AOI_MAX_S,0,1) "
                "is censored identically: once a node passes 60 s its weight is "
                "pinned at 1.0 and further staleness is invisible to the metric.",
        "frac_exposed_at_or_above_cap": sat,
        "by_level": {},
    }
    for lv in sorted({r["_level"] for r in exposed}):
        a = np.array([r["max_aoi_s"] for r in exposed if r["_level"] == lv], dtype=float)
        manip["by_level"][f"{lv:g}"] = {
            "n_exposed": int(a.size),
            "nominal_level_s": float(lv),
            "realised_p50_s": float(np.percentile(a, 50)),
            "realised_p90_s": float(np.percentile(a, 90)),
            "realised_max_s": float(a.max()),
            "frac_at_cap": float(np.mean(a >= cap - 1e-6)),
        }
        e = manip["by_level"][f"{lv:g}"]
        print(f"   nominal {lv:>4g}s -> realised p50 {e['realised_p50_s']:5.1f}s  "
              f"p90 {e['realised_p90_s']:5.1f}s  at-cap {e['frac_at_cap']:.1%}  "
              f"(n={e['n_exposed']})")
    p50s = {k: v["realised_p50_s"] for k, v in manip["by_level"].items()}
    p90s = {k: v["realised_p90_s"] for k, v in manip["by_level"].items()}
    flat = len(set(p50s.values())) == 1 and len(set(p90s.values())) == 1
    manip["realised_p50_identical_across_levels"] = bool(flat)
    manip["verdict"] = (
        "SEVERITY NOT MANIPULATED IN THE MEASURED QUANTITY: realised AoI has "
        f"the same median ({list(p50s.values())[0]:.0f} s) and same p90 at every "
        f"nominal level, and is censored at the {cap:.0f} s cap in {sat:.1%} of "
        "exposed decisions. Tunnel transit time, not the nominal outage "
        "duration, sets the staleness a decision actually sees, and the "
        "observation encoding cannot represent anything past the cap. The "
        "{5,15,30,60} s ladder therefore collapses to a single degraded "
        "condition. Flat DEF and WAMSN across levels is the expected "
        "consequence of an unmanipulated factor and is NOT evidence about "
        "severity; the defensible contrast is binary clean vs degraded."
        if flat else
        f"severity partially manipulated; realised AoI medians differ: {p50s}")
    print(f"   -> {manip['verdict']}")
    out["severity_manipulation_check"] = manip

    # ---- 6. what still needs a re-run
    out["definitive_rerun_required"] = {
        "command": "python scripts/sweep_severity.py "
                   "runs/mappo/B2_gat/central_park_1783964871/ckpt_best.pt "
                   "--seeds 42 43 44 45 46 47 48 49 --episodes 3 "
                   "--faithfulness-every 8 --random-baseline type_matched",
        "why": "type-matched DEF needs fresh counterfactual policy forwards "
               "with composition-matched random sets; it cannot be recovered "
               "from recorded per-decision scalars",
    }

    dest = args.out or (args.sweep_dir / "typematched_ladder.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print()
    print(f"written: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
