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
import copy
import json
from pathlib import Path

import numpy as np


def _numeric(value) -> float:
    """Convert an optional JSON number to float without hiding missing data."""
    return float(value) if value is not None else float("nan")


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
    ranked_x = _rank(x)
    ranked_y = _rank(y)
    ranked_x -= ranked_x.mean()
    ranked_y -= ranked_y.mean()
    denominator = np.sqrt((ranked_x**2).sum() * (ranked_y**2).sum())
    if denominator == 0:
        return 0.0, 1.0
    rho = float((ranked_x * ranked_y).sum() / denominator)
    y_perm = ranked_y.copy()
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(y_perm)
        r = float((ranked_x * y_perm).sum() / denominator)
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
    cell_dir = sweep_dir / "cells"
    candidates = cell_dir.glob("*.json") if cell_dir.is_dir() else sweep_dir.glob("*.json")
    for p in sorted(candidates):
        payload = json.loads(p.read_text())
        if isinstance(payload, dict) and "cell" in payload and "faith_records" in payload:
            cells.append(payload)
    if not cells:
        raise SystemExit(f"no cell JSONs found in {sweep_dir}")
    return manifest, cells


def decisions_frame(cells: list[dict]) -> dict[str, np.ndarray]:
    """Flatten per-decision records across cells into parallel arrays."""
    axis, level, seed, episode = [], [], [], []
    action, pi_full = [], []
    def_, def_m, standard_def, standard_def_m = [], [], [], []
    wamsn, drift, valid_res = [], [], []
    stale_count, stale_share, stale_shift = [], [], []
    paired_def, paired_def_m, paired_def_excl, paired_def_m_excl = [], [], [], []
    record_versions = {
        int(r.get("record_schema_version", 1))
        for c in cells for r in c.get("faith_records", [])
    }
    if len(record_versions) > 1:
        raise ValueError(
            "mixed faithfulness record schemas are not valid statistical input: "
            f"{sorted(record_versions)}"
        )

    for c in cells:
        meta = c["cell"]
        for r in c.get("faith_records", []):
            axis.append(meta["axis"])
            level.append(meta["level"])
            seed.append(meta["seed"])
            episode.append(r.get("episode", 0))
            action.append(r["action"])
            pi_full.append(r.get("pi_full", np.nan))
            standard_def.append(_numeric(r["def"]))
            standard_def_m.append(_numeric(r.get("def_m")))
            def_.append(_numeric(r.get("primary_def", r["def"])))
            def_m.append(_numeric(r.get("primary_def_m", r.get("def_m"))))
            wamsn.append(r["wamsn"])
            drift.append(r.get("drift", np.nan))
            valid_res.append(r["valid_reservations"])
            stale_count.append(r.get("n_stale_veh", 0))
            stale_share.append(r.get("stale_attention_share", np.nan))
            stale_shift.append(r.get("stale_attention_shift", np.nan))
            paired_def.append(_numeric(
                r.get("paired_primary_def_delta", r.get("paired_def_delta"))
            ))
            paired_def_m.append(_numeric(
                r.get("paired_primary_def_m_delta", r.get("paired_def_m_delta"))
            ))
            paired_def_excl.append(_numeric(r.get("paired_def_excl_delta")))
            paired_def_m_excl.append(_numeric(r.get("paired_def_m_excl_delta")))
    return {
        "axis": np.array(axis),
        "level": np.array(level, dtype=np.float64),
        "seed": np.array(seed),
        "episode": np.array(episode, dtype=np.int64),
        "action": np.array(action, dtype=np.int64),
        "pi_full": np.array(pi_full, dtype=np.float64),
        "def": np.array(def_, dtype=np.float64),
        "def_m": np.array(def_m, dtype=np.float64),
        "def_standard": np.array(standard_def, dtype=np.float64),
        "def_m_standard": np.array(standard_def_m, dtype=np.float64),
        "wamsn": np.array(wamsn, dtype=np.float64),
        "drift": np.array(drift, dtype=np.float64),
        "valid_res": np.array(valid_res, dtype=np.int64),
        "stale_count": np.array(stale_count, dtype=np.int64),
        "stale_share": np.array(stale_share, dtype=np.float64),
        "stale_shift": np.array(stale_shift, dtype=np.float64),
        "paired_def_delta": np.array(paired_def, dtype=np.float64),
        "paired_def_m_delta": np.array(paired_def_m, dtype=np.float64),
        "paired_def_excl_delta": np.array(paired_def_excl, dtype=np.float64),
        "paired_def_m_excl_delta": np.array(
            paired_def_m_excl, dtype=np.float64
        ),
    }


_STATISTICAL_FLOAT_FIELDS = {
    "pi_full", "def", "def_m", "def_excl", "def_m_excl",
    "primary_def", "primary_def_m", "primary_g_comp", "primary_g_suff",
    "g_comp", "g_suff", "comp", "suff", "wamsn", "drift",
    "stale_attention_share", "stale_attention_mass", "stale_attention_shift",
    "stale_attention_share_clean_twin", "paired_def_delta",
    "paired_def_m_delta", "paired_def_excl_delta", "paired_def_m_excl_delta",
    "paired_primary_def_delta", "paired_primary_def_m_delta",
}


def quantize_statistical_inputs(cells: list[dict], decimals: int) -> list[dict]:
    """Emulate legacy record quantization without changing the source files."""
    quantized = copy.deepcopy(cells)
    for cell in quantized:
        records = cell.get("faith_records", [])
        for record in records:
            for key in _STATISTICAL_FLOAT_FIELDS:
                value = record.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    record[key] = round(float(value), decimals)

        eligible = [
            record for record in records
            if int(record.get("valid_reservations", 0)) > 0
            and record.get("primary_metric_available", True)
            and record.get("primary_def", record.get("def")) is not None
        ]
        if eligible:
            faithfulness = cell.setdefault("faithfulness", {})
            faithfulness["def_mean"] = float(np.mean([
                record.get("primary_def", record["def"])
                for record in eligible
            ]))
            margin = [
                record.get("primary_def_m", record.get("def_m"))
                for record in eligible
                if record.get("primary_def_m", record.get("def_m")) is not None
            ]
            if margin:
                faithfulness["def_m_mean"] = float(np.mean(margin))
    return quantized


def subset_frame(frame: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    """Return a row-aligned frame subset."""
    return {key: values[mask] for key, values in frame.items()}


def action_stratum_masks(frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Eligible decisions split by whether the policy dispatched a request."""
    eligible = frame["valid_res"] > 0
    return {
        "all": eligible,
        "no_op": eligible & (frame["action"] == 0),
        "dispatch": eligible & (frame["action"] > 0),
    }


def action_stratum_summary(frame: dict[str, np.ndarray]) -> dict:
    """Describe action composition and clean-telemetry faithfulness."""
    clean = frame["axis"] == "clean"
    exposed = (frame["axis"] != "clean") & (frame["stale_count"] > 0)

    def finite_mean(key: str, mask: np.ndarray) -> float | None:
        values = frame[key][mask & np.isfinite(frame[key])]
        return float(values.mean()) if len(values) else None

    return {
        "n_records": int(len(frame["action"])),
        "n_clean_records": int(clean.sum()),
        "n_stale_exposed_records": int(exposed.sum()),
        "mean_selected_action_probability": finite_mean("pi_full", np.ones(len(clean), dtype=bool)),
        "clean_probability_def_mean": finite_mean("def", clean),
        "clean_margin_def_mean": finite_mean("def_m", clean),
    }


def stale_attention_shift_test(
    frame: dict, axis: str, n_permutations: int, n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Episode-cluster test that stale-node attention exceeds its clean twin.

    The endpoint uses a binary stale mask and therefore does not treat the AoI
    value as a causal dose. Only decisions where at least one stale vehicle is
    visible are included.
    """
    mask = ((frame["axis"] == axis) & (frame["stale_count"] > 0)
            & np.isfinite(frame["stale_shift"]))
    exposed = (frame["axis"] == axis) & (frame["stale_count"] > 0)
    if exposed.sum() == 0:
        return {"n_decisions": 0, "note": "no stale-node exposure"}
    if mask.sum() == 0:
        return {"n_decisions": int(exposed.sum()),
                "note": "stale-attention shift unavailable in this legacy sweep"}

    keys = np.array([
        f"{seed}|{level:g}|{episode}" for seed, level, episode in zip(
            frame["seed"][mask], frame["level"][mask], frame["episode"][mask]
        )
    ])
    values = frame["stale_shift"][mask]
    blocks = np.unique(keys)
    block_means = np.array([values[keys == block].mean() for block in blocks])
    lo, hi = bootstrap_ci(block_means, n_boot, rng)
    return {
        "mean_shift": float(block_means.mean()),
        "ci95": [lo, hi],
        "p_one_sided": signflip_p(block_means, n_permutations, rng),
        "n_decisions": int(mask.sum()),
        "n_episode_blocks": int(len(block_means)),
        "interpretation": (
            "positive values mean the degraded observation assigned more "
            "attention mass to stale nodes than its exact clean twin"
        ),
    }


def paired_exposed_def_test(
    frame: dict,
    axis: str,
    metric: str,
    n_permutations: int,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Episode-cluster test of degraded DEF minus exact clean-twin DEF."""
    mask = (
        (frame["axis"] == axis)
        & (frame["stale_count"] > 0)
        & (frame["valid_res"] > 0)
        & np.isfinite(frame[metric])
    )
    if mask.sum() == 0:
        return {"n_decisions": 0, "note": "paired exposed DEF unavailable"}
    keys = np.array([
        f"{seed}|{level:g}|{episode}" for seed, level, episode in zip(
            frame["seed"][mask], frame["level"][mask], frame["episode"][mask]
        )
    ])
    values = frame[metric][mask]
    blocks = np.unique(keys)
    block_means = np.array([values[keys == block].mean() for block in blocks])
    lo, hi = bootstrap_ci(block_means, n_boot, rng)
    return {
        "metric": metric,
        "mean_degraded_minus_clean": float(block_means.mean()),
        "ci95": [lo, hi],
        "p_one_sided_decrease": signflip_p(
            -block_means, n_permutations, rng
        ),
        "n_decisions": int(mask.sum()),
        "n_episode_blocks": int(len(block_means)),
        "interpretation": (
            "negative values mean lower faithfulness on the degraded "
            "observation than on its exact clean twin"
        ),
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
    if metric in ("def", "def_m"):
        m = m & (frame["valid_res"] > 0)  # DEF undefined in no-op-only regime
    m = m & np.isfinite(frame[metric])
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
    def_key: str = "def_mean",
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
        d_clean = clean["faithfulness"].get(def_key)
        p_s = c["mean_pickups"]
        d_s = c["faithfulness"].get(def_key)
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
                "note": "WAMSN constant across decisions (for example, no "
                        "stale nodes attended) — H4 untestable"}
    rho, p = spearman_permutation_p(x, y, "less", n_permutations, rng)
    return {"n": int(len(x)), "rho": rho, "p_one_sided": p}


# ------------------------------------------------- cluster-robust statistics
#
# The pooled per-decision tests above treat ~18k decisions as independent,
# but severity is assigned per (level × seed) cell and decisions cluster
# within episodes. The exchangeable unit under H0 is the episode block, so
# the tests below are the ones a reviewer should trust:
#
#  * restricted permutation — shuffle LEVEL labels across episode blocks
#    (within seed, respecting the crossed design), decisions move with
#    their block;
#  * episode-cluster bootstrap CI on rho — resample whole episodes;
#  * within-episode stratified H4 — correlation computed inside each
#    episode, aggregated across episodes (kills the between-level
#    ecological confound).


def _fast_spearman_levels(level_ranks: np.ndarray, ry_c: np.ndarray) -> float:
    """Spearman when x is already rank-transformed and y-ranks centered."""
    rx = level_ranks - level_ranks.mean()
    denom = np.sqrt((rx**2).sum() * (ry_c**2).sum())
    if denom == 0:
        return 0.0
    return float((rx * ry_c).sum() / denom)


def _level_rank_map(levels: np.ndarray) -> dict[float, float]:
    """Average rank for each distinct level value, given current counts."""
    uniq, counts = np.unique(levels, return_counts=True)
    ranks, start = {}, 0
    for u, c in zip(uniq, counts):
        ranks[float(u)] = start + (c + 1) / 2.0
        start += c
    return ranks


def clustered_h1_h3(
    frame: dict, axis: str, metric: str, alternative: str,
    n_permutations: int, n_boot: int, rng: np.random.Generator,
) -> dict:
    """Episode-block restricted permutation + cluster bootstrap for H1/H3."""
    m = _axis_mask(frame, axis)
    if metric in ("def", "def_m"):
        m = m & (frame["valid_res"] > 0)
    m = m & np.isfinite(frame[metric])
    if m.sum() < 10:
        return {"n": int(m.sum()), "note": "insufficient data"}

    levels = frame["level"][m]
    y = frame[metric][m]
    seeds = frame["seed"][m]
    episodes = frame["episode"][m]

    # Episode blocks: (seed, level, episode) uniquely identifies one episode.
    # Fully vectorised: per permutation only the block→level assignment
    # changes; per-decision level ranks are an O(n) array lookup.
    block_key = np.array([f"{s}|{lv:g}|{e}" for s, lv, e in
                          zip(seeds, levels, episodes)])
    blocks, block_index = np.unique(block_key, return_inverse=True)
    n_blocks = len(blocks)
    first_idx = np.array([np.argmax(block_index == b) for b in range(n_blocks)])
    block_level = levels[first_idx]                      # level per block
    block_seed = seeds[first_idx]                        # seed per block
    uniq_levels = np.unique(levels)
    level_code_of = {float(lv): i for i, lv in enumerate(uniq_levels)}
    block_code = np.array([level_code_of[float(lv)] for lv in block_level])

    ry_c = _rank(y) - _rank(y).mean()
    ry_c_norm = np.sqrt((ry_c**2).sum())

    def _rho_for(block_codes: np.ndarray) -> float:
        codes = block_codes[block_index]                 # per-decision level code
        counts = np.bincount(codes, minlength=len(uniq_levels))
        # average rank per level from cumulative counts
        ends = np.cumsum(counts)
        starts = ends - counts
        rank_of_code = (starts + ends + 1) / 2.0
        rx = rank_of_code[codes]
        rx = rx - rx.mean()
        denom = np.sqrt((rx**2).sum()) * ry_c_norm
        return float((rx * ry_c).sum() / denom) if denom else 0.0

    obs = _rho_for(block_code)

    seed_groups = [np.where(block_seed == s)[0] for s in np.unique(block_seed)]
    perm_code = block_code.copy()
    count = 0
    for _ in range(n_permutations):
        for g in seed_groups:
            perm_code[g] = block_code[g][rng.permutation(len(g))]
        r = _rho_for(perm_code)
        if alternative == "less" and r <= obs:
            count += 1
        elif alternative == "greater" and r >= obs:
            count += 1
    p = (count + 1) / (n_permutations + 1)

    # Episode-cluster bootstrap CI on rho.
    block_members = [np.where(block_index == b)[0] for b in range(n_blocks)]
    rhos = []
    for _ in range(n_boot):
        picked = rng.integers(0, n_blocks, n_blocks)
        idx = np.concatenate([block_members[i] for i in picked])
        rhos.append(spearman(levels[idx], y[idx]))
    ci = (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5)))

    return {"n": int(m.sum()), "n_episode_blocks": n_blocks,
            "rho": obs, "p_episode_perm": float(p),
            "rho_cluster_ci95": ci}


def block_level_trend_test(
    frame: dict, axis: str, metric: str, alternative: str,
    n_permutations: int, n_boot: int, rng: np.random.Generator,
) -> dict:
    """Equal-weight episode-block trend test for action-stratified diagnostics."""
    mask = _axis_mask(frame, axis) & np.isfinite(frame[metric])
    if mask.sum() < 10:
        return {"n": int(mask.sum()), "note": "insufficient data"}

    keys = np.array([
        f"{seed}|{level:g}|{episode}"
        for seed, level, episode in zip(
            frame["seed"][mask], frame["level"][mask], frame["episode"][mask]
        )
    ])
    levels = frame["level"][mask]
    seeds = frame["seed"][mask]
    values = frame[metric][mask]
    blocks = np.unique(keys)
    block_levels = np.array([levels[keys == block][0] for block in blocks])
    block_seeds = np.array([seeds[keys == block][0] for block in blocks])
    block_means = np.array([values[keys == block].mean() for block in blocks])
    if len(blocks) < 10 or len(np.unique(block_levels)) < 2:
        return {"n": int(mask.sum()), "n_episode_blocks": int(len(blocks)),
                "note": "insufficient episode blocks"}

    observed = spearman(block_levels, block_means)
    count = 0
    permuted = block_levels.copy()
    seed_groups = [np.where(block_seeds == seed)[0]
                   for seed in np.unique(block_seeds)]
    for _ in range(n_permutations):
        for group in seed_groups:
            permuted[group] = block_levels[group][rng.permutation(len(group))]
        rho = spearman(permuted, block_means)
        if alternative == "less" and rho <= observed:
            count += 1
        elif alternative == "greater" and rho >= observed:
            count += 1

    boot_rhos = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(blocks), len(blocks))
        boot_rhos.append(spearman(block_levels[picked], block_means[picked]))
    return {
        "n": int(mask.sum()),
        "n_episode_blocks": int(len(blocks)),
        "rho": observed,
        "p_episode_perm": float((count + 1) / (n_permutations + 1)),
        "rho_cluster_ci95": [
            float(np.percentile(boot_rhos, 2.5)),
            float(np.percentile(boot_rhos, 97.5)),
        ],
        "analysis_unit": "equal-weight episode block",
    }


def stratified_h4(
    frame: dict, n_permutations: int, rng: np.random.Generator,
    def_key: str = "def",
) -> dict:
    """H4 with the between-level confound removed: rho(WAMSN, DEF) inside
    each episode, aggregated across episodes; sign-flip test on episode
    rhos (H0: mean rho >= 0)."""
    m = (frame["axis"] != "clean") & (frame["valid_res"] > 0)
    m = m & np.isfinite(frame[def_key])
    seeds = frame["seed"][m]
    levels = frame["level"][m]
    episodes = frame["episode"][m]
    x = frame["wamsn"][m]
    y = frame[def_key][m]
    block_key = np.array([f"{s}|{lv:g}|{e}" for s, lv, e in
                          zip(seeds, levels, episodes)])
    rhos = []
    for b in np.unique(block_key):
        idx = np.where(block_key == b)[0]
        if len(idx) < 8 or np.allclose(x[idx], x[idx][0]):
            continue  # too small or WAMSN constant inside the episode
        rhos.append(spearman(x[idx], y[idx]))
    if len(rhos) < 5:
        return {"n_episodes": len(rhos), "note": "insufficient usable episodes"}
    rhos_arr = np.array(rhos)
    p = signflip_p(-rhos_arr, n_permutations, rng)  # test mean(rho) < 0
    return {"n_episodes": len(rhos_arr),
            "mean_rho_within_episode": float(rhos_arr.mean()),
            "frac_negative": float((rhos_arr < 0).mean()),
            "p_one_sided": float(p)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjusted p-values for one confirmatory family."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    k = len(items)
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (k - i) * p)
        running = max(running, adj)  # enforce monotonicity
        adjusted[name] = running
    return adjusted


def def_agreement(frame: dict) -> dict:
    """Rank agreement between probability-DEF and margin-DEF."""
    m = (frame["valid_res"] > 0) & np.isfinite(frame["def_m"])
    if m.sum() < 10:
        return {"n": int(m.sum()), "note": "insufficient data"}
    return {"n": int(m.sum()),
            "spearman_def_vs_def_m": spearman(frame["def"][m],
                                              frame["def_m"][m])}


# --------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--stat-seed", type=int, default=0,
                        help="RNG seed for permutations/bootstrap only")
    parser.add_argument("--input-round-decimals", type=int,
                        help="sensitivity mode: round statistical records in memory")
    parser.add_argument("--output", type=Path,
                        help="analysis JSON path; defaults to <sweep_dir>/analysis.json")
    args = parser.parse_args()

    manifest, cells = load_sweep(args.sweep_dir)
    if args.input_round_decimals is not None:
        cells = quantize_statistical_inputs(cells, args.input_round_decimals)
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
        m = ((frame["axis"] == ax) & (frame["level"] == lv)
             & (frame["valid_res"] > 0) & np.isfinite(frame["def"]))
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
    results: dict = {
        "schema_version": 2,
        "protocol_version": "2.0",
        "primary_faithfulness_metric": (
            "standard type-matched DEF for no-op; chosen-action-protected "
            "type-matched DEF for dispatch"
        ),
        "input_quantization_decimals": args.input_round_decimals,
        "manifest": {"checkpoint": manifest["checkpoint"],
                     "git_rev": manifest.get("git_rev")},
    }
    has_margin = bool(np.isfinite(frame["def_m"]).any())
    # Axes are discovered from the sweep itself (old sweeps: dropout_rate /
    # tunnel_noise; ladder sweeps: outage_duration / dropout_outage_duration).
    axes = sorted({c["cell"]["axis"] for c in cells} - {"clean"})
    for axis in axes:
        results[f"H1_{axis}"] = hypothesis_h1_h3(
            frame, axis, "def", "less", args.n_permutations, rng)
        results[f"H2_{axis}"] = hypothesis_h2(
            cells, axis, args.n_permutations, rng)
        results[f"H3_{axis}"] = hypothesis_h1_h3(
            frame, axis, "wamsn", "greater", args.n_permutations, rng)
        if has_margin:
            results[f"H1m_{axis}"] = hypothesis_h1_h3(
                frame, axis, "def_m", "less", args.n_permutations, rng)
            results[f"H2m_{axis}"] = hypothesis_h2(
                cells, axis, args.n_permutations, rng, def_key="def_m_mean")
    results["H4_pooled"] = hypothesis_h4(frame, args.n_permutations, rng)

    def _verdict(r: dict, label: str) -> str:
        if "note" in r:
            return f"{label}: n/a ({r['note']}, n={r['n']})"
        p = r.get("p_one_sided")
        stat = r.get("rho", r.get("mean_delta_faith_minus_perf"))
        sig = "SUPPORTED" if p is not None and p < 0.05 else "not supported"
        return f"{label}: {sig}  (stat={stat:+.3f}, p={p:.4f}, n={r['n']})"

    print("hypothesis tests (α = 0.05, one-sided):")
    for axis in axes:
        print(" ", _verdict(results[f"H1_{axis}"], f"H1 (DEF ↓ with {axis})"))
        if f"H1m_{axis}" in results:
            print(" ", _verdict(results[f"H1m_{axis}"], f"H1m (margin-DEF ↓ with {axis})"))
        print(" ", _verdict(results[f"H2_{axis}"], f"H2 (faith declines faster, {axis})"))
        if f"H2m_{axis}" in results:
            print(" ", _verdict(results[f"H2m_{axis}"], f"H2m (margin-faith declines faster, {axis})"))
        print(" ", _verdict(results[f"H3_{axis}"], f"H3 (WAMSN ↑ with {axis})"))
    print(" ", _verdict(results["H4_pooled"], "H4 (WAMSN–DEF negative, pooled)"))

    # ---- cluster-robust section (the statistics a reviewer should trust)
    primary_axis = ("outage_duration" if "outage_duration" in axes
                    else "max_aoi" if "max_aoi" in axes else axes[0])
    robust: dict = {"primary_axis": primary_axis}
    print()
    print("cluster-robust statistics (episode-block permutation, primary):")
    attention_shift = stale_attention_shift_test(
        frame, primary_axis, args.n_permutations, args.n_bootstrap, rng
    )
    robust["primary_stale_attention_shift"] = attention_shift
    if "note" in attention_shift:
        print(f"  stale-attention shift: n/a ({attention_shift['note']})")
    else:
        print(
            "  stale-attention shift (binary stale mask, degraded - clean twin): "
            f"mean={attention_shift['mean_shift']:+.4f}  "
            f"CI95=[{attention_shift['ci95'][0]:+.4f}, "
            f"{attention_shift['ci95'][1]:+.4f}]  "
            f"p={attention_shift['p_one_sided']:.4f}  "
            f"(episode blocks={attention_shift['n_episode_blocks']})"
        )
    paired_followup = {}
    for label, metric in (
        ("probability_def", "paired_def_delta"),
        ("margin_def", "paired_def_m_delta"),
        ("protected_probability_def", "paired_def_excl_delta"),
        ("protected_margin_def", "paired_def_m_excl_delta"),
    ):
        paired = paired_exposed_def_test(
            frame, primary_axis, metric, args.n_permutations,
            args.n_bootstrap, rng
        )
        paired_followup[label] = paired
        if "note" not in paired:
            print(
                f"  paired exposed {label}: "
                f"delta={paired['mean_degraded_minus_clean']:+.4f}  "
                f"CI95=[{paired['ci95'][0]:+.4f}, {paired['ci95'][1]:+.4f}]  "
                f"p(decrease)={paired['p_one_sided_decrease']:.4f}  "
                f"(blocks={paired['n_episode_blocks']})"
            )
    robust["exposure_conditioned_paired_followup"] = paired_followup
    for label, metric, alt in (("H1_robust", "def", "less"),
                               ("H1m_robust", "def_m", "less"),
                               ("H3_robust", "wamsn", "greater")):
        if metric == "def_m" and not has_margin:
            continue
        r = clustered_h1_h3(frame, primary_axis, metric, alt,
                            args.n_permutations, args.n_bootstrap, rng)
        robust[label] = r
        if "note" in r:
            print(f"  {label}: n/a ({r['note']})")
        else:
            print(f"  {label} ({metric} vs {primary_axis}): rho={r['rho']:+.3f}  "
                  f"p_episode={r['p_episode_perm']:.4f}  "
                  f"CI95=[{r['rho_cluster_ci95'][0]:+.3f}, {r['rho_cluster_ci95'][1]:+.3f}]  "
                  f"(blocks={r['n_episode_blocks']})")
    for label, key in (("H4_within_episode", "def"),
                       ("H4m_within_episode", "def_m")):
        if key == "def_m" and not has_margin:
            continue
        r = stratified_h4(frame, args.n_permutations, rng, def_key=key)
        robust[label] = r
        if "note" in r:
            print(f"  {label}: n/a ({r['note']})")
        else:
            print(f"  {label}: mean within-episode rho={r['mean_rho_within_episode']:+.3f}  "
                  f"({r['frac_negative']:.0%} episodes negative)  "
                  f"p={r['p_one_sided']:.4f}  (episodes={r['n_episodes']})")

    # ---- confirmatory family (proposal-preregistered H1-H4, probability-DEF,
    # primary axis) with Holm correction; margin variants and secondary axes
    # are exploratory by declaration.
    family = {}
    for name, key in (("H1", f"H1_{primary_axis}"), ("H2", f"H2_{primary_axis}"),
                      ("H3", f"H3_{primary_axis}")):
        p = results.get(key, {}).get("p_one_sided")
        if p is not None:
            family[name] = p
    if "p_one_sided" in results.get("H4_pooled", {}):
        family["H4"] = results["H4_pooled"]["p_one_sided"]
    # Prefer the robust p-values where computed — they are the primary tests.
    if "H1_robust" in robust and "p_episode_perm" in robust["H1_robust"]:
        family["H1"] = robust["H1_robust"]["p_episode_perm"]
    if "H3_robust" in robust and "p_episode_perm" in robust["H3_robust"]:
        family["H3"] = robust["H3_robust"]["p_episode_perm"]
    if "H4_within_episode" in robust and "p_one_sided" in robust["H4_within_episode"]:
        family["H4"] = robust["H4_within_episode"]["p_one_sided"]
    adj = holm(family)
    robust["confirmatory_family_raw_p"] = family
    robust["confirmatory_family_holm_p"] = adj
    print()
    print("confirmatory family (robust p where available) with Holm correction:")
    for name in sorted(family):
        verdict = "SUPPORTED" if adj[name] < 0.05 else "not supported"
        print(f"  {name}: raw p={family[name]:.4f} → Holm p={adj[name]:.4f}  {verdict}")
    print("  (margin-DEF variants and secondary axes are exploratory by "
          "declaration — reported unadjusted, flagged as such)")

    robust["def_agreement"] = def_agreement(frame)
    da = robust["def_agreement"]
    if "spearman_def_vs_def_m" in da:
        print(f"\nprob-DEF vs margin-DEF rank agreement: "
              f"rho={da['spearman_def_vs_def_m']:+.3f} (n={da['n']})")

    # Action strata are a declared diagnostic because request dispatches and
    # no-op decisions have different operational meanings and frequencies.
    masks = action_stratum_masks(frame)
    eligible_count = int(masks["all"].sum())
    action_strata = {}
    print("\naction-stratified diagnostics (exploratory):")
    for name, mask in masks.items():
        stratum = subset_frame(frame, mask)
        summary = action_stratum_summary(stratum)
        summary["fraction_of_eligible_records"] = (
            float(mask.sum() / eligible_count) if eligible_count else 0.0
        )
        summary["primary_stale_attention_shift"] = stale_attention_shift_test(
            stratum, primary_axis, args.n_permutations, args.n_bootstrap, rng
        )
        summary["paired_probability_def"] = paired_exposed_def_test(
            stratum, primary_axis, "paired_def_delta",
            args.n_permutations, args.n_bootstrap, rng
        )
        summary["paired_margin_def"] = paired_exposed_def_test(
            stratum, primary_axis, "paired_def_m_delta",
            args.n_permutations, args.n_bootstrap, rng
        )
        summary["H1_probability_def"] = block_level_trend_test(
            stratum, primary_axis, "def", "less",
            args.n_permutations, args.n_bootstrap, rng
        )
        summary["H1_margin_def"] = block_level_trend_test(
            stratum, primary_axis, "def_m", "less",
            args.n_permutations, args.n_bootstrap, rng
        )
        summary["H4_probability_def"] = stratified_h4(
            stratum, args.n_permutations, rng, def_key="def"
        )
        summary["H4_margin_def"] = stratified_h4(
            stratum, args.n_permutations, rng, def_key="def_m"
        )
        action_strata[name] = summary
        paired = summary["paired_probability_def"]
        paired_value = paired.get("mean_degraded_minus_clean")
        paired_text = "n/a" if paired_value is None else f"{paired_value:+.6f}"
        print(
            f"  {name:<8} n={summary['n_records']:<6} "
            f"({summary['fraction_of_eligible_records']:.1%})  "
            f"mean pi={summary['mean_selected_action_probability']:.4f}  "
            f"paired DEF={paired_text}"
        )
    robust["action_strata"] = action_strata
    robust["action_strata_interpretation"] = (
        "Exploratory diagnostic. The all-decision result is split into no-op "
        "and request-dispatch decisions without changing the frozen policy or "
        "the predeclared primary hypothesis family."
    )
    results["robust"] = robust

    out_path = args.output or (args.sweep_dir / "analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print()
    print(f"analysis: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
