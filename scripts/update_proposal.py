"""Apply the agreed architecture-alignment edits to the research proposal.

Three targeted text substitutions in paragraph 42 ("Fleet environment"), each
matching a specific gap we identified between the proposal and the
implementation. Paragraph 42 is a single run so `run.text = ...` is safe and
preserves surrounding formatting.

Kept as a repeatable script (rather than one-shot Python-in-shell) so the
edits are auditable and re-runnable if the proposal is re-imported later.

Run once:
    python scripts/update_proposal.py

The backup is preserved at
docs/Research_Proposal_Faithfulness_Decoupling.backup.docx.
"""
from __future__ import annotations

import sys
from pathlib import Path

from docx import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = PROJECT_ROOT / "docs" / "Research_Proposal_Faithfulness_Decoupling.docx"


REPLACEMENTS = [
    # Difference 1: drop zone-based repositioning; keep the implemented
    # "Discrete(K+1) — no-op or accept one of the K nearest reservations".
    (
        "The city is partitioned into zones defining the repositioning "
        "action space; idle vehicles are agents that each step stay, "
        "reposition toward demand, or accept an assigned order.",
        "Idle vehicles are agents that each step either stay idle or accept "
        "one of the K nearest pending reservations, giving an action space "
        "of Discrete(K+1). No explicit zone partition or learned "
        "repositioning is used; idle taxis default to the SUMO taxi "
        "device's randomCircling behaviour and become available for "
        "dispatch again on the next step.",
    ),
    # Difference 2: keep position, velocity, availability, AoI; drop the
    # remaining-capacity and assigned-tasks fields (not carried by the env).
    (
        "Each vehicle carries features: position, velocity, availability, "
        "remaining capacity, assigned tasks and Age of Information (AoI).",
        "Each vehicle carries features: position, velocity, availability "
        "and Age of Information (AoI).",
    ),
    # Difference 5: rewrite the implementation-stack sentence to reflect
    # the hand-rolled GAT (chosen for direct access to attention weights)
    # rather than PyTorch Geometric.
    (
        "Implementation uses Python, SUMO/libsumo, PyTorch, PyTorch "
        "Geometric and PettingZoo; a lightweight grid simulator "
        "(Lin et al. [1]) is kept as a fast fallback if SUMO stepping "
        "is too slow.",
        "Implementation uses Python, SUMO driven through the in-process "
        "libsumo backend, PyTorch and PettingZoo. The graph attention "
        "layer is a hand-rolled multi-head scaled dot-product attention "
        "rather than PyTorch Geometric's GATConv, so that attention "
        "weights are first-class return values of the forward pass and "
        "can be consumed directly by the faithfulness pipeline. A "
        "lightweight grid simulator (Lin et al. [1]) is kept as a fast "
        "fallback if SUMO stepping is too slow.",
    ),
    # Difference 3a: introduce the decentralised fleet-dispatch framing
    # before describing the per-vehicle action space.
    (
        "so the dispatch layer need not be rebuilt. Idle vehicles are agents",
        "so the dispatch layer need not be rebuilt. We adopt a "
        "decentralised fleet-dispatch formulation [1, 5]: rather than a "
        "central dispatcher computing a global matching each step, every "
        "idle vehicle chooses independently whether to accept one of its "
        "K nearest pending reservations. Idle vehicles are agents",
    ),
    # Difference 3b: replace the vehicle-only graph description with the
    # per-agent heterogeneous graph the code actually implements.
    (
        "Graph: nodes are vehicles; edges encode distance, travel time "
        "and task relationship.",
        "Per-agent heterogeneous graph. For each acting vehicle, a small "
        "graph is constructed with three node types: self, K nearest "
        "neighbouring vehicles (to model fleet coordination) and K "
        "nearest pending reservations (the candidate assignments). Edges "
        "are implicit — the GAT layer attends over all valid nodes with "
        "a masked softmax, and edge weights are learned rather than "
        "pre-computed. Relative distance is provided as a node feature "
        "(Δx, Δy, ‖Δ‖ from self) rather than as an explicit edge attribute.",
    ),
    # Difference 3c: add per-decision faithfulness rationale AND turn the
    # decentralised-dispatch conflict weakness into a research question
    # that motivates DEF.
    (
        "and can be consumed directly by the faithfulness pipeline. "
        "A lightweight grid simulator",
        "and can be consumed directly by the faithfulness pipeline. This "
        "per-agent, decentralised graph is chosen deliberately: attention "
        "rows are then in one-to-one correspondence with a single "
        "vehicle's dispatch decision, allowing DEF and WAMSN to be "
        "evaluated on each decision independently rather than on an "
        "aggregate dispatcher output; AoI is likewise a per-agent "
        "quantity, matching the summands in the WAMSN definition "
        "(Section 7.4). A known drawback of decentralised dispatch is "
        "that multiple vehicles can converge on the same reservation "
        "without explicit coordination; the GAT's attention over "
        "neighbouring vehicles gives each agent implicit awareness of "
        "peers likely to compete for the same order, allowing "
        "coordination to emerge from the learned policy — whether that "
        "coordination signal is faithfully explained by attention is "
        "itself one of the questions DEF is designed to answer. "
        "A lightweight grid simulator",
    ),
    # Remove the "lightweight grid simulator fallback" line from the fleet-
    # environment description (para 42). The feature was described but
    # never implemented; keeping the sentence would misrepresent scope.
    (
        "answer. A lightweight grid simulator (Lin et al. [1]) is kept "
        "as a fast fallback if SUMO stepping is too slow.",
        "answer.",
    ),
    # And remove the "falls back to the lightweight grid simulator" clause
    # from the Risk-mitigation list (para 97). The other mitigations
    # (reused MAPPO, reduced fleet size, prioritising DEF/WAMSN) still
    # stand.
    (
        "reduces fleet size and model complexity, falls back to the "
        "lightweight grid simulator (§7.1), and prioritises",
        "reduces fleet size and model complexity, and prioritises",
    ),
    # §7.7 fix (1/3): actor description — replace the GATv2 + zone-based
    # action-space fragment with the hand-rolled multi-head attention +
    # Discrete(K+1) formulation that the code actually implements.
    (
        "a GATv2 encoder over the vehicle graph feeding an action head "
        "over {stay, reposition to a neighbour zone, accept order} with "
        "invalid-action masking",
        "a hand-rolled multi-head attention encoder over the per-agent "
        "heterogeneous graph (self + K nearest neighbouring vehicles + "
        "K nearest pending reservations) feeding an action head of "
        "Discrete(K+1) — either stay idle or accept one of the K "
        "candidate reservations — with invalid-reservation masking",
    ),
    # §7.7 fix (2/3): reward shaping — drop the "small penalty for empty
    # repositioning" clause since repositioning is not an action, and
    # replace with the actual pickup / dispatch / wait triple used in
    # env.py (see paragraph 42).
    (
        "Reward shaping encourages cooperation: positive for orders "
        "served, negative for waiting time, small penalty for empty "
        "repositioning.",
        "Reward shaping encourages cooperation: positive per completed "
        "pickup and per successful dispatch attempt, negative in "
        "proportion to the mean pending-rider waiting time. No "
        "repositioning penalty is used because repositioning is not "
        "part of the action space.",
    ),
    # §7.7 fix (3/3): implementation stack — replace the "reused MAPPO
    # library + PyTorch Geometric GAT encoder" line with the reality
    # (hand-rolled MAPPO + hand-rolled GAT, motivated by the need for
    # first-class attention access in §7.4). Kept as a rationale sentence
    # rather than a bare stack list to preserve the paragraph's flow.
    (
        "PPO is reused from a maintained MAPPO library with a PyTorch "
        "Geometric GAT encoder, not reimplemented.",
        "The MAPPO training loop and the multi-head attention encoder "
        "are implemented from scratch in `src/dispatch_marl/` so that "
        "attention weights remain first-class outputs of the policy's "
        "forward pass — a requirement for the faithfulness pipeline of "
        "§7.4 rather than a design preference.",
    ),
    # §7.8 fix: drop torch-geometric from the Colab install list — we
    # don't depend on it (see §7.7 above). Everything else in the list
    # stays.
    (
        "eclipse-sumo, libsumo, traci, torch, torch-geometric, "
        "pettingzoo, pandas, pyarrow, numpy, matplotlib",
        "eclipse-sumo, libsumo, traci, torch, pettingzoo, pandas, "
        "pyarrow, numpy, matplotlib",
    ),
    # ------------------------------------------------------------------
    # July 2026 batch — methodology-description alignment after the
    # freeze-mechanism rework. Confirmed with the author that the
    # proposal is not yet frozen for assessment. Deliberately touches
    # NO hypotheses, expectations, contributions or timeline (results
    # must be judged against the ex-ante predictions).
    # ------------------------------------------------------------------
    # J1a (§7.2): tunnel transit time is fixed by geography once the map
    # is chosen, so the severity level is operationalised as an outage
    # duration (receiver re-acquisition delay) that bounds max AoI.
    (
        "This grounds staleness geographically and produces bursty, "
        "realistic degradation, with severity controlled by the maximum "
        "AoI reached.",
        "This grounds staleness geographically and produces bursty, "
        "realistic degradation, with severity controlled by the maximum "
        "AoI reached. Because tunnel transit time is fixed by network "
        "geography once the map is chosen, each severity level is "
        "operationalised as a signal-outage duration: after the trigger, "
        "the signal stays lost until the vehicle's AoI reaches the level "
        "(physically, receiver re-acquisition delay), so the level bounds "
        "the maximum AoI directly while long transits are reported via "
        "the empirical AoI distribution.",
    ),
    # J1b (§7.2 table header): align the parenthetical with J1a.
    (
        "Maximum AoI (tunnel transit duration)",
        "Maximum AoI (signal-outage duration)",
    ),
    # J2a (§7.6): degradation is applied live at the observation boundary
    # with an exact per-decision clean twin — a strictly stronger paired
    # design than degrading logged episodes offline.
    (
        "Apply the AoI staleness operator (7.2) — freezing telemetry on "
        "tunnel edges — to the clean test logs at each severity, keeping "
        "the clean copy aligned for paired comparison.",
        "Apply the AoI staleness operator (7.2) — freezing telemetry on "
        "tunnel edges — live at the observation boundary at each "
        "severity; because the simulator state itself is never "
        "corrupted, every degraded observation has an exact clean twin "
        "generated in the same step, giving paired clean/degraded "
        "comparisons per decision while also capturing the policy's "
        "closed-loop behaviour under degradation (which offline replay "
        "of logged episodes cannot).",
    ),
    # J2b (§7.6): artefact format — versioned JSON manifests, matching
    # tunnels.json / demand_manifest.json / per-sweep manifest.json.
    (
        "Store the SUMO network/config, splits and settings as "
        "compressed .npz/Parquet with a manifest (seed, parameters, "
        "hash) so any result regenerates exactly.",
        "Store the SUMO network/config, demand variants and their "
        "chronological splits, and per-sweep settings as versioned JSON "
        "manifests (seed, parameters, git revision) so any result "
        "regenerates exactly.",
    ),
    # J3a (§7.5): trim the performance-metric list to what the study
    # reports (completed pickups, pending-rider wait, episode reward).
    (
        "Metrics span dispatch performance (average response time, "
        "fleet utilisation, task-completion rate, episode reward),",
        "Metrics span dispatch performance (completed pickups per "
        "episode, mean pending-rider waiting time, episode reward),",
    ),
    # J3b (§7.4): drop the binary-AMSN robustness check — graded WAMSN
    # is used throughout and the check was cut in the simplification.
    (
        "approaching 1 as it concentrates on maximally stale ones (a "
        "binary AMSN with a fixed threshold is reported as a robustness "
        "check).",
        "approaching 1 as it concentrates on maximally stale ones.",
    ),
]


def main() -> int:
    if not DOC_PATH.exists():
        print(f"ERROR: {DOC_PATH} not found", file=sys.stderr)
        return 1

    doc = Document(DOC_PATH)

    # Concatenate all paragraph text so we can search for each snippet
    # without hard-coding paragraph indices (defends against future re-imports
    # that renumber paragraphs). Tables (e.g. the §7.2 severity ladder) are
    # walked too — their text lives outside doc.paragraphs.
    all_paragraphs = list(doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                all_paragraphs.extend(cell.paragraphs)

    hits = 0
    for pi, para in enumerate(all_paragraphs):
        for old, new in REPLACEMENTS:
            if old not in para.text:
                continue
            # Idempotency guard: several NEW texts *contain* their OLD
            # text (pure insertions), so a re-run would match again and
            # duplicate the insertion. Skip if NEW is already in place.
            if new in para.text:
                continue
            # First try a run-local edit — if OLD is fully contained inside
            # one run, we can modify only that run's text and preserve every
            # other run's formatting (bold labels, italics, hyperlinks, ...).
            applied_in_run = False
            for run in para.runs:
                if old in run.text:
                    run.text = run.text.replace(old, new)
                    applied_in_run = True
                    print(f"[para {pi}] applied in-run replacement ({len(old)} → {len(new)} chars)")
                    break
            if not applied_in_run:
                # OLD spans multiple runs — no clean way to keep sub-run
                # formatting. Fall back to join-and-replace with a warning.
                print(f"[para {pi}] WARNING: OLD text spans {len(para.runs)} runs; "
                      "fell back to join-and-replace (sub-run formatting may be lost)")
                joined = "".join(r.text for r in para.runs).replace(old, new)
                for r in para.runs[1:]:
                    r.text = ""
                para.runs[0].text = joined
            hits += 1

    if hits == 0:
        print("ERROR: no replacement matched. The proposal text has drifted "
              "from what this script expects; open the doc and inspect.",
              file=sys.stderr)
        return 1

    doc.save(DOC_PATH)
    print(f"\napplied {hits} replacement(s); saved: {DOC_PATH}")
    print("backup: docs/Research_Proposal_Faithfulness_Decoupling.backup.docx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
