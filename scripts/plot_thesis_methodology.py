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
    fig, ax = canvas("Graph-Attention Policy and Read-Only Audit", (16, 8.8))

    # The upper band shows the model's operational computation.
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.49), 0.96, 0.36,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.0, edgecolor="#CAD1D6", facecolor="#FAFBFC",
    ))
    ax.text(0.04, 0.81, "POLICY COMPUTATION", ha="left", va="center",
            fontsize=11.5, fontweight="bold", color=GREY)

    box(ax, 0.05, 0.56, 0.16, 0.19, "Local graph",
        "self taxi, peer taxis\nand requests", BLUE,
        title_size=13, body_size=10.5)
    box(ax, 0.27, 0.56, 0.17, 0.19, "Typed encoder",
        "project features and\nadd node-type embedding", GREY,
        title_size=13, body_size=10.2)
    box(ax, 0.50, 0.56, 0.19, 0.19, "Attention encoder",
        "2 scaled dot-product layers\n4 heads per layer", "#6741A5",
        title_size=13, body_size=10.5)
    box(ax, 0.77, 0.68, 0.18, 0.10, "Actor",
        "action logits", GREEN, title_size=12.5, body_size=10.2)
    box(ax, 0.77, 0.53, 0.18, 0.10, "Critic (training only)",
        "team value", BLUE, title_size=11.8, body_size=10.2)

    arrow(ax, (0.21, 0.655), (0.27, 0.655))
    arrow(ax, (0.44, 0.655), (0.50, 0.655))
    arrow(ax, (0.69, 0.655), (0.77, 0.73), color=GREEN)
    arrow(ax, (0.69, 0.655), (0.77, 0.58), color=BLUE)

    # The lower band makes the audit reduction explicit and separate.
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.11), 0.96, 0.30,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.2, edgecolor=ORANGE, facecolor="#FFF9F5",
    ))
    ax.text(0.04, 0.37, "READ-ONLY EXPLANATION AUDIT", ha="left", va="center",
            fontsize=11.5, fontweight="bold", color=ORANGE)

    box(ax, 0.25, 0.17, 0.18, 0.14, "Attention tensor",
        r"$A^{(l,h)} \in \mathbb{R}^{N \times N}$", ORANGE,
        title_size=12.5, body_size=11)
    box(ax, 0.49, 0.17, 0.18, 0.14, "Self-node row",
        r"$A^{(l,h)}_{0,:}$", ORANGE,
        title_size=12.5, body_size=11)
    box(ax, 0.73, 0.17, 0.22, 0.14, "Node importance",
        "mean over layers and heads\nused by DEF and WAMSN", ORANGE,
        title_size=12.5, body_size=10.2)

    audit_arrow = FancyArrowPatch(
        (0.595, 0.56), (0.34, 0.31), arrowstyle="-|>",
        connectionstyle="angle3,angleA=-90,angleB=180",
        mutation_scale=16, linewidth=1.7, color=ORANGE,
    )
    ax.add_patch(audit_arrow)
    arrow(ax, (0.43, 0.24), (0.49, 0.24), color=ORANGE)
    arrow(ax, (0.67, 0.24), (0.73, 0.24), color=ORANGE)

    ax.text(0.04, 0.075,
            "The audit reads attention already produced by the policy; it does not "
            "change the actor, critic or selected action.",
            ha="left", va="center", fontsize=10.8, color="#202020",
            fontweight="bold")
    save(fig, "simplified_graph_attention_design")


def observation_action_graph():
    fig, ax = canvas("Local Observation Graph for One Idle Taxi", (16, 8.2))

    nodes = {
        "self": (0.31, 0.49),
        "taxi 1": (0.13, 0.72),
        "taxi 2": (0.10, 0.43),
        "taxi 3": (0.18, 0.18),
        "request 1": (0.52, 0.72),
        "request 2": (0.58, 0.47),
        "request 3": (0.50, 0.19),
    }

    # Valid nodes are fully connected. The darker arrows highlight the
    # self-node attention row exposed as the explanation.
    node_items = list(nodes.items())
    for index, (_, start) in enumerate(node_items):
        for _, end in node_items[index + 1:]:
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color="#B8BEC4", linewidth=0.75, alpha=0.32, zorder=1)

    attention_widths = {
        "taxi 1": 1.2,
        "taxi 2": 3.6,
        "taxi 3": 1.8,
        "request 1": 2.8,
        "request 2": 1.4,
        "request 3": 2.2,
    }
    for name, width in attention_widths.items():
        ax.add_patch(FancyArrowPatch(
            nodes[name], nodes["self"], arrowstyle="-|>", mutation_scale=14,
            linewidth=width, color="#343A40", alpha=0.82,
            shrinkA=28, shrinkB=31, zorder=2,
        ))

    def graph_node(name, label, colour, size=3000):
        x, y = nodes[name]
        ax.scatter([x], [y], s=size, color=colour, edgecolor="white",
                   linewidth=2.5, zorder=4)
        ax.text(x, y, label, ha="center", va="center", fontsize=10.5,
                color="white", fontweight="bold", zorder=5,
                linespacing=1.15)

    graph_node("self", "SELF\nT0", BLUE, size=3900)
    for index in range(1, 4):
        graph_node(f"taxi {index}", f"PEER\nT{index}", GREEN)
    for index in range(1, 4):
        graph_node(f"request {index}", f"REQ\nR{index}", ORANGE)

    stale_x, stale_y = nodes["taxi 2"]
    ax.scatter([stale_x], [stale_y], s=3800, facecolors="none",
               edgecolors="#B02A37", linewidth=3.0, zorder=6)
    ax.text(stale_x - 0.005, stale_y - 0.095, "stale: AoI = 30 s",
            ha="center", va="top", fontsize=10, color="#B02A37",
            fontweight="bold")

    ax.text(0.12, 0.84, "Peer-taxi context", ha="center", fontsize=12,
            color=GREEN, fontweight="bold")
    ax.text(0.52, 0.84, "Passenger requests", ha="center", fontsize=12,
            color=ORANGE, fontweight="bold")
    ax.text(0.31, 0.60, "attention into\nthe acting taxi",
            ha="center", va="center", fontsize=9.5, color="#202020",
            fontweight="bold", zorder=7)

    box(ax, 0.76, 0.25, 0.21, 0.53, "Actor action set",
        "action 0   no-op\naction 1   serve R1\n"
        "action 2   serve R2\naction 3   serve R3",
        BLUE, body_size=10.5)
    action_y = {"request 1": 0.47, "request 2": 0.43, "request 3": 0.39}
    for name, target_y in action_y.items():
        ax.add_patch(FancyArrowPatch(
            nodes[name], (0.76, target_y), arrowstyle="-|>",
            mutation_scale=14, linewidth=1.8, color=ORANGE,
            linestyle="--", shrinkA=28, shrinkB=4, zorder=3,
            connectionstyle="arc3,rad=0.05",
        ))

    ax.text(0.34, 0.075,
            "Valid nodes are fully connected; padded nodes are masked.  "
            "Line width shows relative attention into the self taxi.",
            ha="center", va="center", fontsize=10.5, color="#303030")
    ax.text(0.865, 0.14,
            "A request node defines an action.\nA peer taxi only supplies context.",
            ha="center", va="center", fontsize=10.5, color="#202020",
            fontweight="bold")
    save(fig, "observation_graph_action_mapping")


def model_training_design():
    fig, ax = canvas("Model Training and Leakage-Controlled Evaluation", (16, 9.2))

    ax.add_patch(FancyBboxPatch(
        (0.02, 0.66), 0.96, 0.21,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.0, edgecolor="#CAD1D6", facecolor="#FAFBFC",
    ))
    ax.text(0.04, 0.835, "MODEL CONDITIONS", ha="left", va="center",
            fontsize=11.5, fontweight="bold", color=GREY)
    models = [
        ("MLP", "MAPPO\nclean", GREY),
        ("GAT", "GAT-MAPPO\nclean", GREEN),
        ("GAT-NoAoI", "GAT-MAPPO\nclean, no AoI", BLUE),
        ("GAT-Outage", "GAT-MAPPO\n30-s outage", ORANGE),
    ]
    model_xs = (0.05, 0.285, 0.52, 0.755)
    for x, (model, description, color) in zip(model_xs, models):
        box(ax, x, 0.69, 0.19, 0.105, model, description, color,
            title_size=12.5, body_size=9.4)

    ax.add_patch(FancyBboxPatch(
        (0.02, 0.32), 0.96, 0.28,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.0, edgecolor="#CAD1D6", facecolor="#FAFBFC",
    ))
    ax.text(0.04, 0.565, "CHECKPOINT CONTROL", ha="left", va="center",
            fontsize=11.5, fontweight="bold", color=GREY)

    pipeline = [
        (0.145, "Train all four", "150 epochs\nseeds 42, 43, 44\nsave every 10 epochs", BLUE),
        (0.38, "Validate and select",
         "validation demand\npickups first; reward tie-break", GREEN),
        (0.615, "Freeze", "one checkpoint\nper model and seed", GREY),
        (0.85, "Open held-out test", "evaluate once\nno reselection", ORANGE),
    ]
    width, y, height = 0.18, 0.37, 0.15
    for x, title, body, color in pipeline:
        box(ax, x - width / 2, y, width, height, title, body, color,
            title_size=12.3, body_size=9.4)
    for left, right in zip(pipeline, pipeline[1:]):
        arrow(ax, (left[0] + width / 2, 0.445),
              (right[0] - width / 2, 0.445))

    ax.text(0.5, 0.285, "HELD-OUT EVALUATION SCOPE", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=ORANGE)
    box(ax, 0.18, 0.105, 0.28, 0.13, "Clean comparison",
        "All four models\n8 evaluation seeds x 3 episodes", BLUE,
        title_size=12.5, body_size=9.6)
    box(ax, 0.57, 0.105, 0.31, 0.13, "Faithfulness audit",
        "GAT versus GAT-Outage\nclean + 10, 20, 30 and 60 s outages", ORANGE,
        title_size=12.5, body_size=9.6)

    for target_x, color, curvature in ((0.32, BLUE, 0.18),
                                       (0.725, ORANGE, -0.12)):
        branch = FancyArrowPatch(
            (0.85, 0.37), (target_x, 0.235), arrowstyle="-|>",
            connectionstyle=f"arc3,rad={curvature}",
            mutation_scale=16, linewidth=1.6, color=color,
        )
        ax.add_patch(branch)

    ax.text(0.5, 0.055,
            "Validation selects the checkpoint; the held-out test never changes it.",
            ha="center", fontsize=10.8, fontweight="bold", color="#202020")
    save(fig, "model_design_training_process")

def telemetry_flow():
    fig, ax = canvas("Tunnel Trigger and Observation-Layer Degradation", (16, 8.8))

    ax.add_patch(FancyBboxPatch(
        (0.02, 0.11), 0.68, 0.75,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.1, edgecolor="#CAD1D6", facecolor="#FAFBFC",
    ))
    ax.add_patch(FancyBboxPatch(
        (0.73, 0.11), 0.25, 0.75,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.2, edgecolor=GREY, facecolor="#F8F8F8",
    ))
    ax.text(0.04, 0.82, "OPERATIONAL SIMULATION", ha="left", va="center",
            fontsize=12, fontweight="bold", color=BLUE)
    ax.text(0.855, 0.82, "READ-ONLY AUDIT", ha="center", va="center",
            fontsize=12, fontweight="bold", color=GREY)

    lane_specs = [
        (0.695, "SUMO\nGROUND TRUTH", BLUE),
        (0.455, "OBSERVATION\nLAYER", ORANGE),
        (0.215, "POLICY", GREEN),
    ]
    for y, label, color in lane_specs:
        ax.text(0.045, y, label, ha="left", va="center", fontsize=9.7,
                fontweight="bold", color=color, linespacing=1.15)

    box(ax, 0.16, 0.63, 0.14, 0.13, "Live state",
        "position, speed, edge", BLUE, title_size=11.7, body_size=9.1)
    box(ax, 0.36, 0.63, 0.14, 0.13, "Tunnel entry",
        "outage trigger", ORANGE, title_size=11.4, body_size=9.1)
    box(ax, 0.56, 0.63, 0.12, 0.13, "State advances",
        "SUMO stays live", BLUE, title_size=10.9, body_size=8.9)
    arrow(ax, (0.30, 0.695), (0.36, 0.695), color=BLUE)
    arrow(ax, (0.50, 0.695), (0.56, 0.695), color=BLUE)

    box(ax, 0.16, 0.39, 0.14, 0.13, "Current telemetry",
        "latest reading", ORANGE, title_size=11.2, body_size=9.1)
    stale_patch = FancyBboxPatch(
        (0.36, 0.39), 0.14, 0.13,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.7, edgecolor=ORANGE, facecolor="#FFF3E8",
        hatch="///", zorder=2,
    )
    ax.add_patch(stale_patch)
    ax.text(0.43, 0.485, "Last update held", ha="center", va="center",
            fontsize=11.2, fontweight="bold", color=ORANGE, zorder=3,
            bbox=dict(facecolor="#FFF9F5", edgecolor="none", pad=1.2))
    ax.text(0.43, 0.425, "x, y, speed frozen\nAoI increases",
            ha="center", va="center", fontsize=8.7, color="#202020", zorder=3,
            linespacing=1.15,
            bbox=dict(facecolor="#FFF9F5", edgecolor="none", pad=1.0))
    box(ax, 0.56, 0.39, 0.12, 0.13, "Telemetry resumes",
        "current reading", GREEN, title_size=10.5, body_size=8.8)
    arrow(ax, (0.30, 0.455), (0.36, 0.455), color=ORANGE)
    arrow(ax, (0.50, 0.455), (0.56, 0.455), color=GREEN)

    arrow(ax, (0.23, 0.63), (0.23, 0.52), color=BLUE)
    arrow(ax, (0.43, 0.63), (0.43, 0.52), color=ORANGE)
    arrow(ax, (0.62, 0.63), (0.62, 0.52), color=GREEN)

    box(ax, 0.34, 0.15, 0.18, 0.13, "Observed graph",
        "contains stale taxi data", GREEN, title_size=11.5, body_size=9.1)
    box(ax, 0.56, 0.15, 0.12, 0.13, "Policy action",
        "live decision", GREEN, title_size=10.8, body_size=8.9)
    arrow(ax, (0.43, 0.39), (0.43, 0.28), color=GREEN)
    arrow(ax, (0.52, 0.215), (0.56, 0.215), color=GREEN)

    ax.plot([0.68, 0.70], [0.215, 0.215], color=GREEN, linewidth=1.7)
    ax.plot([0.70, 0.70], [0.215, 0.695], color=GREEN, linewidth=1.7)
    arrow(ax, (0.70, 0.695), (0.68, 0.695), color=GREEN)
    ax.text(0.695, 0.35, "action to SUMO", ha="right", va="center",
            rotation=90, fontsize=8.8, color=GREEN, fontweight="bold")

    audit_steps = [
        (0.64, "Same SUMO step", "true current state"),
        (0.44, "Clean observation", "no telemetry freeze"),
        (0.24, "Paired comparison", "clean versus degraded"),
    ]
    for y, title, body in audit_steps:
        box(ax, 0.77, y, 0.17, 0.11, title, body, GREY,
            title_size=10.8, body_size=8.9)
    arrow(ax, (0.855, 0.64), (0.855, 0.55), color=GREY, dashed=True)
    arrow(ax, (0.855, 0.44), (0.855, 0.35), color=GREY, dashed=True)
    ax.text(0.855, 0.155, "No action is sent to SUMO.",
            ha="center", va="center", fontsize=9.4, color=GREY,
            fontweight="bold")

    ax.add_patch(FancyBboxPatch(
        (0.04, 0.045), 0.025, 0.022,
        boxstyle="square,pad=0", linewidth=1.2, edgecolor=ORANGE,
        facecolor="#FFF3E8", hatch="///",
    ))
    ax.text(0.075, 0.056, "frozen observation", ha="left", va="center",
            fontsize=9.8, color="#202020")
    ax.plot([0.27, 0.31], [0.056, 0.056], color=GREEN, linewidth=1.8)
    ax.text(0.32, 0.056, "operational path", ha="left", va="center",
            fontsize=9.8, color="#202020")
    ax.plot([0.50, 0.54], [0.056, 0.056], color=GREY, linewidth=1.5,
            linestyle="--")
    ax.text(0.55, 0.056, "audit-only path", ha="left", va="center",
            fontsize=9.8, color="#202020")
    ax.text(0.98, 0.056, "Trigger in SUMO; degradation in observation layer.",
            ha="right", va="center", fontsize=10.2, fontweight="bold",
            color="#202020")
    save(fig, "telemetry_degradation_data_flow")


def construct_validity_design():
    fig, ax = canvas("Why Node Occlusion Needs Construct-Validity Controls", (16, 8.8))

    panels = [
        (0.02, 0.49, 0.46, 0.37, "#FFF9F5", ORANGE,
         "REQUEST-NODE OCCLUSION"),
        (0.52, 0.49, 0.46, 0.37, "#F7FBF7", GREEN,
         "PEER-TAXI OCCLUSION"),
    ]
    for x, y, width, height, fill, color, title in panels:
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.2, edgecolor=color, facecolor=fill,
        ))
        ax.text(x + 0.02, y + height - 0.055, title, ha="left", va="center",
                fontsize=11.5, fontweight="bold", color=color)

    def crossed_node(x, y, label, color):
        node = Circle((x, y), 0.046, facecolor=color, edgecolor="white",
                      linewidth=2.2, zorder=3)
        ax.add_patch(node)
        ax.text(x, y, label, ha="center", va="center", fontsize=12,
                fontweight="bold", color="white", zorder=4)
        ax.plot([x - 0.055, x + 0.055], [y + 0.055, y - 0.055],
                color="#A33A3A", linewidth=3.0, zorder=5)

    crossed_node(0.105, 0.66, "Rk", ORANGE)
    ax.text(0.105, 0.565, "hide request", ha="center", va="center",
            fontsize=9.8, fontweight="bold", color=ORANGE)
    arrow(ax, (0.16, 0.66), (0.21, 0.66), color=ORANGE)

    request_effect = FancyBboxPatch(
        (0.21, 0.565), 0.23, 0.19,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.7, edgecolor=ORANGE, facecolor="#FFF3E8",
        hatch="///",
    )
    ax.add_patch(request_effect)
    ax.text(0.325, 0.695, "Action set changes", ha="center", va="center",
            fontsize=12.2, fontweight="bold", color=ORANGE,
            bbox=dict(facecolor="#FFF9F5", edgecolor="none", pad=1.2))
    ax.text(0.325, 0.625, "serve Rk disappears", ha="center", va="center",
            fontsize=10.2, color="#202020",
            bbox=dict(facecolor="#FFF9F5", edgecolor="none", pad=1.0))
    ax.text(0.325, 0.535, "Structural confound", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#A33A3A")

    crossed_node(0.605, 0.66, "Tk", GREEN)
    ax.text(0.605, 0.565, "hide peer taxi", ha="center", va="center",
            fontsize=9.8, fontweight="bold", color=GREEN)
    arrow(ax, (0.66, 0.66), (0.71, 0.66), color=GREEN)
    box(ax, 0.71, 0.565, 0.23, 0.19, "Action set unchanged",
        "all request actions remain", GREEN, title_size=12.2, body_size=10.2)
    ax.text(0.825, 0.535, "Context removed only", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=GREEN)

    ax.add_patch(FancyBboxPatch(
        (0.02, 0.10), 0.96, 0.31,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.1, edgecolor=BLUE, facecolor="#F7FAFC",
    ))
    ax.text(0.04, 0.375, "CORRECTED DEF COMPARISON", ha="left", va="center",
            fontsize=11.5, fontweight="bold", color=BLUE)

    box(ax, 0.06, 0.17, 0.33, 0.13, "Compare top-k subsets",
        "attention-ranked versus random-ranked", ORANGE,
        title_size=11.7, body_size=9.5)
    box(ax, 0.46, 0.145, 0.32, 0.18, "Match the intervention",
        "same k and node-type mix\nprotect chosen request; log clamps",
        BLUE, title_size=11.7, body_size=9.4)
    box(ax, 0.84, 0.17, 0.12, 0.13, "DEF",
        "attention\nminus random", GREEN, title_size=12, body_size=9.1)

    arrow(ax, (0.39, 0.235), (0.46, 0.235), color=ORANGE)
    arrow(ax, (0.78, 0.235), (0.84, 0.235), color=BLUE)

    ax.text(0.5, 0.055,
            "The random baseline must create the same kind of perturbation as the explanation subset.",
            ha="center", va="center", fontsize=10.5, color="#202020",
            fontweight="bold")
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
