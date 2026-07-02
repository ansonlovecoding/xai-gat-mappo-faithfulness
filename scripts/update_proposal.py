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
]


def main() -> int:
    if not DOC_PATH.exists():
        print(f"ERROR: {DOC_PATH} not found", file=sys.stderr)
        return 1

    doc = Document(DOC_PATH)

    # Concatenate all paragraph text so we can search for each snippet
    # without hard-coding paragraph indices (defends against future re-imports
    # that renumber paragraphs).
    hits = 0
    for pi, para in enumerate(doc.paragraphs):
        for old, new in REPLACEMENTS:
            if old in para.text:
                # Single-run edit is the safest path — para.text setter
                # wipes formatting; direct run-text substitution preserves it.
                if len(para.runs) == 1:
                    para.runs[0].text = para.runs[0].text.replace(old, new)
                    print(f"[para {pi}] applied replacement ({len(old)} → {len(new)} chars)")
                    hits += 1
                else:
                    # Multi-run paragraph: replace text and warn about
                    # potential formatting loss.
                    print(f"[para {pi}] WARNING: {len(para.runs)} runs; "
                          "replacement may lose sub-run formatting")
                    joined = "".join(r.text for r in para.runs)
                    joined = joined.replace(old, new)
                    # Wipe all but the first run; put the new text into the first.
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
