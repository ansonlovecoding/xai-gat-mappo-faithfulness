"""Create concise methodology diagrams that match the dissertation v4 code."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
BLUE = "#176B87"
GREEN = "#3A7D44"
ORANGE = "#C75000"
GREY = "#555555"
LIGHT = "#F5F7F8"


def box(ax, x, y, width, height, title, body, color=BLUE, title_size=12,
        body_size=10, align="center"):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.5, edgecolor=color, facecolor=LIGHT,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.76, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=color)
    ax.text(x + (width / 2 if align == "center" else width * 0.07),
            y + height * 0.38, body,
            ha=align, va="center", fontsize=body_size, color="#202020",
            linespacing=1.35)
    return patch


def arrow(ax, start, end, color="#202020", style="-|>", dashed=False):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=16,
                            linewidth=1.5, color=color,
                            linestyle="--" if dashed else "-")
    ax.add_patch(patch)
    return patch


def canvas(title, size=(16, 7.5)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.96, title, ha="center", va="top", fontsize=23,
            fontweight="bold")
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def graph_attention_design():
    fig, ax = canvas("Graph-Attention Policy and Explanation Channel")
    xs = [0.025, 0.275, 0.525, 0.775]
    width, height, y = 0.20, 0.66, 0.18
    box(ax, xs[0], y, width, height, "1  Decision graph",
        "Self taxi\n+ up to 5 peer taxis\n+ up to 5 requests\n\nOne graph per idle taxi",
        BLUE, body_size=11)
    box(ax, xs[1], y, width, height, "2  Typed embeddings",
        "Self, taxi and request\nfeatures use separate projections\n\nType embedding added\nShared width: 64",
        GREY, body_size=10.5)
    box(ax, xs[2], y, width, height, "3  Graph self-attention",
        "Fully connected valid nodes\nPadding keys are masked\n\n2 layers x 4 heads\n\nsoftmax(QK^T / sqrt(d_h))",
        "#6741A5", body_size=10.5)
    box(ax, xs[3], y, width, height, "4  Two outputs",
        "Policy output\nNo-op + request-action logits\n\nExplanation output\nSelf-node attention row\naggregated across layers/heads",
        GREEN, body_size=10.2)
    for left, right in zip(xs, xs[1:]):
        arrow(ax, (left + width, y + height / 2), (right, y + height / 2))
    ax.text(0.5, 0.09,
            "Request nodes are also action candidates; peer-taxi nodes only provide context. "
            "This asymmetry requires type-matched faithfulness controls.",
            ha="center", va="center", fontsize=11, color=ORANGE,
            fontweight="bold")
    save(fig, "simplified_graph_attention_design")


def observation_action_graph():
    fig, ax = canvas("Observation Graph and Request-to-Action Mapping", (16, 7.8))

    box(ax, 0.035, 0.58, 0.22, 0.22, "Self taxi",
        "position, time, speed\nAoI, position-valid flag",
        BLUE, body_size=10.5)
    box(ax, 0.035, 0.31, 0.22, 0.22, "Up to 5 peer taxis",
        "relative position, availability\ndistance and AoI",
        GREEN, body_size=10.5)
    box(ax, 0.035, 0.04, 0.22, 0.22, "Up to 5 requests",
        "pickup/drop-off vectors\nand waiting time",
        ORANGE, body_size=10.5)

    box(ax, 0.34, 0.26, 0.27, 0.45, "Valid-node graph",
        "All valid nodes can exchange\ninformation through masked\ngraph self-attention\n\nPadding nodes are excluded",
        "#6741A5", body_size=11)
    for y in (0.69, 0.42, 0.15):
        arrow(ax, (0.255, y), (0.34, 0.49))

    box(ax, 0.69, 0.48, 0.27, 0.27, "Actor action set",
        "action 0: no-op\nactions 1-5: visible requests",
        BLUE, body_size=11)
    arrow(ax, (0.61, 0.49), (0.69, 0.61))
    arrow(ax, (0.255, 0.15), (0.69, 0.55), color=ORANGE, dashed=True)
    ax.text(0.48, 0.13, "each request node supplies one candidate action",
            ha="center", fontsize=10, color=ORANGE, fontweight="bold")

    ax.text(0.825, 0.32,
            "Removing a request can remove an action.\n"
            "Removing a peer taxi only hides context.",
            ha="center", va="center", fontsize=11, color="#202020",
            fontweight="bold")
    save(fig, "observation_graph_action_mapping")


def model_training_design():
    fig, ax = canvas("Model Conditions, Training, Selection and Evaluation", (16, 8.2))
    # Model conditions
    ax.text(0.18, 0.86, "Model conditions", ha="center", fontsize=15,
            fontweight="bold")
    models = [
        ("B1", "MLP-MAPPO\nclean training", GREY),
        ("B2", "GAT-MAPPO\nclean training", GREEN),
        ("B3", "GAT-MAPPO, no AoI\nclean training", BLUE),
        ("H5", "GAT-MAPPO\n30-s outage training", ORANGE),
    ]
    for index, (model, text, color) in enumerate(models):
        y = 0.70 - index * 0.145
        box(ax, 0.035, y, 0.29, 0.105, model, text, color,
            title_size=13, body_size=9.5)

    box(ax, 0.39, 0.58, 0.22, 0.23, "Train each condition",
        "150 epochs\ntraining seeds 42, 43, 44\ncheckpoints every 10 epochs",
        BLUE, body_size=11)
    box(ax, 0.39, 0.27, 0.22, 0.23, "Validation selection",
        "3 stochastic validation episodes\nseed 2026\nrank by mean pickups\nthen reward, then earlier epoch",
        GREEN, body_size=9.8)
    arrow(ax, (0.325, 0.52), (0.39, 0.69))
    arrow(ax, (0.50, 0.58), (0.50, 0.50))

    box(ax, 0.675, 0.58, 0.29, 0.23, "Freeze selected checkpoint",
        "Test split remains untouched during selection\n\nOne selected policy per model and seed",
        GREY, body_size=10.2)
    box(ax, 0.675, 0.27, 0.29, 0.23, "Held-out evaluation",
        "8 test-demand seeds x 3 episodes\nClean + 10, 20, 30, 60-s outages\n\nPickups, DEF, WAMSN, paired stale shift",
        ORANGE, body_size=10.1)
    arrow(ax, (0.61, 0.69), (0.675, 0.69))
    arrow(ax, (0.82, 0.58), (0.82, 0.50))
    ax.text(0.5, 0.10,
            "Faithfulness sweeps run only for B2 and H5 after checkpoint selection.",
            ha="center", fontsize=11, fontweight="bold", color="#202020")
    save(fig, "model_design_training_process")


def telemetry_flow():
    fig, ax = canvas("Tunnel-Triggered Degradation at the Observation Boundary", (16, 8.2))
    ax.text(0.17, 0.86, "SUMO ground truth", ha="center", fontsize=15,
            fontweight="bold", color=BLUE)
    ax.text(0.51, 0.86, "Observation boundary", ha="center", fontsize=15,
            fontweight="bold", color=ORANGE)
    ax.text(0.84, 0.86, "Policy and evaluation", ha="center", fontsize=15,
            fontweight="bold", color=GREEN)

    box(ax, 0.035, 0.48, 0.24, 0.26, "True traffic state",
        "SUMO updates position, speed\nand current road edge every step\n\nThe simulator state is never frozen",
        BLUE, body_size=10.5)
    box(ax, 0.34, 0.60, 0.27, 0.16, "Tunnel-entry event?",
        "Trigger only on outside-to-inside transition\nRemaining inside does not retrigger",
        ORANGE, body_size=9.7)
    box(ax, 0.34, 0.31, 0.27, 0.22, "Fixed observation outage",
        "Serve last trusted x, y and speed\nAoI = now - last trusted time\nWindow: 10, 20, 30 or 60 s\nThen accept a fresh reading",
        ORANGE, body_size=9.5)
    arrow(ax, (0.275, 0.61), (0.34, 0.68))
    arrow(ax, (0.475, 0.60), (0.475, 0.53))

    box(ax, 0.68, 0.60, 0.27, 0.16, "Observed graph",
        "Self, peer-taxi and request nodes\nPolicy acts on the observed state",
        GREEN, body_size=10)
    box(ax, 0.68, 0.31, 0.27, 0.22, "Paired evaluation",
        "Degraded observation\nversus exact clean twin\n\nDEF, WAMSN and stale-attention shift",
        GREEN, body_size=10)
    arrow(ax, (0.61, 0.68), (0.68, 0.68))
    arrow(ax, (0.61, 0.42), (0.68, 0.42))
    arrow(ax, (0.815, 0.60), (0.815, 0.53))
    arrow(ax, (0.815, 0.31), (0.17, 0.31), color=GREY, dashed=True)
    ax.text(0.49, 0.27, "dispatch action returns to the real SUMO world",
            ha="center", fontsize=10, color=GREY)
    ax.text(0.5, 0.10,
            "The tunnel supplies the trigger; the observation layer creates stale telemetry. "
            "A taxi must leave and re-enter before another tunnel trigger.",
            ha="center", fontsize=11, fontweight="bold", color="#202020")
    save(fig, "telemetry_degradation_data_flow")


def construct_validity_design():
    fig, ax = canvas("Why the Faithfulness Audit Needs Matched Interventions", (16, 7.8))

    box(ax, 0.035, 0.48, 0.27, 0.30, "Request-node occlusion",
        "Request Rk is hidden\nIts corresponding action k disappears\nThe available decision set changes",
        ORANGE, body_size=11)
    box(ax, 0.365, 0.48, 0.27, 0.30, "Peer-taxi occlusion",
        "Taxi Tk is hidden\nAll request actions remain available\nOnly contextual information changes",
        GREEN, body_size=11)
    box(ax, 0.695, 0.48, 0.27, 0.30, "Unmatched random control",
        "Can contain a different mixture\nof request and taxi nodes\nDEF may measure intervention type",
        "#A33A3A", body_size=11)
    arrow(ax, (0.305, 0.63), (0.365, 0.63), color=GREY)
    arrow(ax, (0.635, 0.63), (0.695, 0.63), color=GREY)

    box(ax, 0.18, 0.13, 0.64, 0.23, "Corrected v4 comparison",
        "Match the number and type of occluded nodes\n"
        "Protect the chosen request-action   |   Log margin clamps",
        BLUE, body_size=10.5)
    arrow(ax, (0.83, 0.48), (0.70, 0.36), color=BLUE)
    ax.text(0.5, 0.06,
            "The corrected DEF asks whether attention ranks nodes better than a fair, structurally equivalent random control.",
            ha="center", fontsize=11, color="#202020", fontweight="bold")
    save(fig, "construct_validity_action_deletion")


def main():
    observation_action_graph()
    graph_attention_design()
    model_training_design()
    telemetry_flow()
    construct_validity_design()
    print(f"methodology figures: {OUT}")


if __name__ == "__main__":
    main()
