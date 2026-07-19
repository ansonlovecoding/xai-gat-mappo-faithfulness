"""Construct-validity audit of DEF (P2): how much of the negative score
is the occlusion-= action-deletion mechanism, and what does WAMSN mean
conditionally?

Reads an instrumented sweep (sweep_severity.py run after the P2
instrumentation) and reports:

  1. **Clamp audit** — fraction of counterfactual forwards whose margin hit
     ±cap, split by top-k vs random occlusion sets. A large random-set
     excess means the random baseline is inflated by draws that delete the
     chosen action.
  2. **Exclusion-variant DEF** — DEF with the chosen reservation's node
     protected from occlusion on both sides. If def_m_excl ≈ 0 while
     def_m ≪ 0, the "below random" finding is carried by the mechanical
     asymmetry, not by information relevance.
  3. **Conditional WAMSN** — exposure rate (decisions with ≥1 stale vehicle
     node visible), WAMSN among exposed decisions, stale-in-top3 rate, and
     top-3 churn vs the clean twin: the operator-facing translation.
  4. **Empirical AoI distribution per ladder level** (proposal §7.2's
     honesty requirement: geography can exceed the level).

Usage:
  python scripts/audit_construct_validity.py runs/sweeps/<instrumented-sweep>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()

    cells = []
    for p in sorted(args.sweep_dir.glob("*.json")):
        if p.name in ("manifest.json", "analysis.json", "audit.json"):
            continue
        cells.append(json.loads(p.read_text()))

    recs = []
    for c in cells:
        for r in c.get("faith_records", []):
            r["_axis"] = c["cell"]["axis"]
            r["_level"] = c["cell"]["level"]
            recs.append(r)
    inst = [r for r in recs if "clamp_topk" in r]
    if not inst:
        raise SystemExit("no instrumented records — rerun the sweep after P2")
    scored = [r for r in inst if r["valid_reservations"] > 0]
    excl = [r for r in scored if "def_m_excl" in r]

    out: dict = {"n_records": len(inst), "n_scored": len(scored),
                 "n_excl_evaluated": len(excl)}

    # ---- 1. clamp audit
    ct = np.array([r["clamp_topk"] for r in scored])
    cr = np.array([r["clamp_rand"] for r in scored])
    out["clamp_topk_mean"] = float(ct.mean())
    out["clamp_rand_mean"] = float(cr.mean())
    print(f"records: {len(inst)}  scored: {len(scored)}  "
          f"exclusion-evaluated: {len(excl)}")
    print()
    print("1. CLAMP AUDIT (fraction of counterfactual forwards hitting ±cap)")
    print(f"   top-k occlusion sets:  {ct.mean():.3f}")
    print(f"   random baseline sets:  {cr.mean():.3f}")

    # ---- 2. exclusion-variant verdict
    dm = np.array([r["def_m"] for r in excl])
    dmx = np.array([r["def_m_excl"] for r in excl])
    dp = np.array([r["def"] for r in excl])
    dpx = np.array([r["def_excl"] for r in excl])
    out["def_m_mean"] = float(dm.mean())
    out["def_m_excl_mean"] = float(dmx.mean())
    out["def_mean"] = float(dp.mean())
    out["def_excl_mean"] = float(dpx.mean())
    # bootstrap CI on the exclusion-variant mean
    rng = np.random.default_rng(0)
    boots = [dmx[rng.integers(0, len(dmx), len(dmx))].mean() for _ in range(2000)]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    out["def_m_excl_ci95"] = ci
    print()
    print("2. EXCLUSION-VARIANT DEF (chosen reservation's node protected)")
    print(f"   margin-DEF  standard: {dm.mean():+.4f}   "
          f"exclusion: {dmx.mean():+.4f}  CI95 [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"   prob-DEF    standard: {dp.mean():+.4f}   exclusion: {dpx.mean():+.4f}")
    mech_share = 1.0 - (abs(dmx.mean()) / max(abs(dm.mean()), 1e-9))
    out["mechanical_share_of_def_m"] = float(mech_share)
    verdict = ("ARTIFACT-DOMINATED: the below-random DEF is carried by the "
               "action-deletion mechanism" if abs(dmx.mean()) < 0.1 * abs(dm.mean())
               else "MIXED: exclusion shrinks but does not eliminate the effect"
               if abs(dmx.mean()) < 0.5 * abs(dm.mean())
               else "ROBUST: the negative DEF survives exclusion")
    out["verdict"] = verdict
    print(f"   → {verdict}")
    print(f"     ({mech_share:.0%} of the standard margin-DEF magnitude "
          "disappears when the chosen node is protected)")

    # ---- 3. conditional WAMSN / operator translation
    stale_present = np.array([r.get("n_stale_veh", 0) > 0 for r in inst])
    wamsn = np.array([r["wamsn"] for r in inst])
    deg = [r for r, s in zip(inst, stale_present) if s]
    out["stale_exposure_rate"] = float(stale_present.mean())
    print()
    print("3. CONDITIONAL WAMSN (operator translation)")
    print(f"   stale-node exposure rate: {stale_present.mean():.3f} of decisions")
    if deg:
        w = np.array([r["wamsn"] for r in deg])
        t3 = np.array([bool(r.get("stale_in_top3")) for r in deg])
        out["wamsn_conditional_mean"] = float(w.mean())
        out["stale_in_top3_rate_conditional"] = float(t3.mean())
        print(f"   WAMSN | exposed:          {w.mean():.3f} "
              f"(vs pooled {wamsn.mean():.3f})")
        print(f"   stale node in top-3 | exposed: {t3.mean():.1%} of decisions")
    churns = np.array([r["top3_churn"] for r in inst if "top3_churn" in r])
    if churns.size:
        out["top3_churn_mean"] = float(churns.mean())
        out["top3_churn_any_rate"] = float((churns > 0).mean())
        print(f"   top-3 vs clean twin: mean {churns.mean():.2f} of 3 slots "
              f"change; ≥1 slot changes in {(churns > 0).mean():.1%} of decisions")

    # ---- 4. empirical AoI per ladder level
    print()
    print("4. EMPIRICAL AoI PER LADDER LEVEL (seconds, exposed decisions)")
    by_level: dict = {}
    for r in inst:
        if r.get("n_stale_veh", 0) > 0 and r["_axis"] != "clean":
            by_level.setdefault(r["_level"], []).append(r.get("max_aoi_s", 0.0))
    for lv in sorted(by_level):
        a = np.array(by_level[lv])
        by = {"n": int(len(a)), "p50": float(np.percentile(a, 50)),
              "p90": float(np.percentile(a, 90)), "max": float(a.max())}
        out.setdefault("aoi_by_level", {})[f"{lv:g}"] = by
        print(f"   level {lv:>4g}s: median {by['p50']:>5.1f}  "
              f"p90 {by['p90']:>6.1f}  max {by['max']:>6.1f}  (n={by['n']})")

    (args.sweep_dir / "audit.json").write_text(json.dumps(out, indent=2))
    print(f"\naudit: {args.sweep_dir / 'audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
