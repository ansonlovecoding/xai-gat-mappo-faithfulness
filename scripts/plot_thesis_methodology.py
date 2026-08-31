"""Create concise methodology diagrams that match the dissertation v4 code."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge


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
        ("GAT-Outage", "GAT-MAPPO\n30-s outage", ORANGE),
    ]
    model_xs = (0.11, 0.405, 0.70)
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
        (0.145, "Train all three", "150 epochs\nseeds 42, 43, 44\nsave every 10 epochs", BLUE),
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
    box(ax, 0.16, 0.105, 0.32, 0.13, "Clean-telemetry capability",
        "MLP, GAT and GAT-Outage\nper checkpoint: 8 seeds x 3 episodes = 24", BLUE,
        title_size=12.1, body_size=9.2)
    box(ax, 0.55, 0.105, 0.35, 0.13, "Faithfulness audit",
        "GAT and GAT-Outage; frozen checkpoints\nclean + 10, 20, 30 and 60 s outages", ORANGE,
        title_size=12.5, body_size=9.2)

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
    fig, ax = canvas("Tunnel-Triggered Telemetry Degradation", (16, 8.8))

    def taxi(x, y, color, alpha=1.0, scale=1.0, outline=False):
        width, height = 0.065 * scale, 0.035 * scale
        face = "white" if outline else color
        linestyle = "--" if outline else "-"
        ax.add_patch(Rectangle(
            (x - width / 2, y - height / 2), width, height,
            linewidth=1.4, edgecolor=color, facecolor=face,
            linestyle=linestyle, alpha=alpha, zorder=8,
        ))
        roof = Polygon([
            (x - width * 0.26, y + height / 2),
            (x - width * 0.12, y + height * 0.92),
            (x + width * 0.20, y + height * 0.92),
            (x + width * 0.34, y + height / 2),
        ], closed=True, linewidth=1.3, edgecolor=color, facecolor=face,
           linestyle=linestyle, alpha=alpha, zorder=8)
        ax.add_patch(roof)
        for wheel_x in (x - width * 0.27, x + width * 0.27):
            ax.add_patch(Circle((wheel_x, y - height * 0.55),
                                0.007 * scale, facecolor="#202020",
                                edgecolor="white", linewidth=0.7,
                                alpha=alpha, zorder=9))
        if not outline:
            ax.text(x, y, "TAXI", ha="center", va="center", fontsize=7.2 * scale,
                    color="white", fontweight="bold", alpha=alpha, zorder=10)

    def signal(x, y, color, crossed=False):
        ax.add_patch(Circle((x, y), 0.006, facecolor=color,
                            edgecolor="none", zorder=10))
        for radius in (0.018, 0.031, 0.044):
            ax.add_patch(Arc((x, y), radius * 2, radius * 2,
                             theta1=35, theta2=145, linewidth=1.5,
                             color=color, zorder=10))
        if crossed:
            ax.plot([x - 0.037, x + 0.037], [y + 0.04, y - 0.025],
                    color="#B02A37", linewidth=2.5, zorder=11)

    # Road scene: tunnel entry is the physical trigger, but SUMO motion continues.
    ax.add_patch(Rectangle((0.03, 0.58), 0.94, 0.24,
                           facecolor="#EDF6F8", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.03, 0.58), 0.94, 0.105,
                           facecolor="#484D50", edgecolor="none", zorder=1))
    for start in [0.06, 0.18, 0.30, 0.68, 0.80, 0.92]:
        ax.plot([start, min(start + 0.055, 0.97)], [0.632, 0.632],
                color="white", linewidth=2.0, alpha=0.85, zorder=2)

    tunnel_x, tunnel_y = 0.50, 0.60
    ax.add_patch(Wedge((tunnel_x, tunnel_y), 0.165, 0, 180,
                       facecolor="#7D858A", edgecolor="#555555",
                       linewidth=2.0, zorder=3))
    ax.add_patch(Wedge((tunnel_x, tunnel_y), 0.118, 0, 180,
                       facecolor="#25292C", edgecolor="#B9C0C4",
                       linewidth=1.2, zorder=4))
    ax.add_patch(Rectangle((tunnel_x - 0.165, 0.58), 0.33, 0.052,
                           facecolor="#7D858A", edgecolor="#555555",
                           linewidth=1.5, zorder=3))
    ax.add_patch(Rectangle((tunnel_x - 0.118, 0.58), 0.236, 0.052,
                           facecolor="#25292C", edgecolor="none", zorder=5))
    ax.text(tunnel_x, 0.79, "TUNNEL", ha="center", va="center",
            fontsize=12.5, color="#303030", fontweight="bold")

    taxi(0.26, 0.64, BLUE, scale=0.92)
    taxi(0.49, 0.64, "#8DB7C4", alpha=0.45, scale=0.92)
    taxi(0.75, 0.64, GREEN, scale=0.92)
    arrow(ax, (0.30, 0.64), (0.42, 0.64), color=BLUE)
    arrow(ax, (0.57, 0.64), (0.70, 0.64), color=GREEN)
    signal(0.26, 0.73, BLUE)
    signal(0.43, 0.73, ORANGE, crossed=True)

    ax.axvline(0.38, ymin=0.58, ymax=0.82, color=ORANGE,
               linewidth=2.0, linestyle="--", zorder=7)
    ax.text(0.38, 0.845, "Tunnel entry", ha="center", va="bottom",
            fontsize=11.3, color=ORANGE, fontweight="bold")
    ax.text(0.38, 0.805, "starts outage timer", ha="center", va="bottom",
            fontsize=9.2, color="#202020")
    ax.text(0.75, 0.72, "SUMO vehicle keeps moving", ha="center", va="center",
            fontsize=10.5, color=GREEN, fontweight="bold")

    # Timeline labels and shared time markers.
    x0, x1, x2 = 0.28, 0.56, 0.79
    ax.text(0.045, 0.45, "SUMO TRUE STATE", ha="left", va="center",
            fontsize=11.5, color=BLUE, fontweight="bold")
    ax.text(0.045, 0.27, "POLICY OBSERVATION", ha="left", va="center",
            fontsize=11.5, color=ORANGE, fontweight="bold")
    ax.text(0.045, 0.105, "READ-ONLY CLEAN TWIN", ha="left", va="center",
            fontsize=10.8, color=GREY, fontweight="bold")
    for x, label in ((x0, "t0"), (x1, "during outage"), (x2, "t0 + T")):
        ax.plot([x, x], [0.08, 0.50], color="#D5DADD", linewidth=0.8,
                linestyle=":" if x != x0 else "--", zorder=0)
        ax.text(x, 0.515, label, ha="center", va="bottom", fontsize=9.2,
                color="#404040", fontweight="bold" if x == x0 else "normal")

    # SUMO path continues through the complete outage window.
    arrow(ax, (0.22, 0.45), (0.86, 0.45), color=BLUE)
    taxi(x0, 0.45, BLUE, scale=0.72)
    taxi(x1, 0.45, BLUE, scale=0.72)
    taxi(x2, 0.45, BLUE, scale=0.72)
    ax.text(0.88, 0.45, "live", ha="left", va="center", fontsize=9.4,
            color=BLUE, fontweight="bold")

    # The policy-side observation is pinned to the last valid location.
    ax.plot([0.22, x0], [0.27, 0.27], color=ORANGE, linewidth=1.8)
    ax.plot([x0, x2], [0.27, 0.27], color=ORANGE, linewidth=5.5,
            alpha=0.22, solid_capstyle="round")
    taxi(x0, 0.27, ORANGE, scale=0.78)
    taxi(x1, 0.27, ORANGE, alpha=0.35, scale=0.78, outline=True)
    ax.plot([x1 - 0.025, x1 + 0.025], [0.295, 0.245],
            color="#B02A37", linewidth=2.0)
    ax.plot([x1 - 0.025, x1 + 0.025], [0.245, 0.295],
            color="#B02A37", linewidth=2.0)
    taxi(x2, 0.27, GREEN, scale=0.78)
    ax.text((x0 + x2) / 2, 0.335, "last valid position held; AoI increases",
            ha="center", va="center", fontsize=10.2, color=ORANGE,
            fontweight="bold")
    ax.text(x0, 0.205, "AoI = 0", ha="center", fontsize=9.1, color="#303030")
    ax.text(x2, 0.205, "current reading resumes", ha="center", fontsize=9.1,
            color=GREEN, fontweight="bold")

    # The clean twin reads the same SUMO step for comparison but never acts.
    ax.plot([0.22, 0.85], [0.105, 0.105], color=GREY, linewidth=1.4,
            linestyle="--")
    taxi(x1, 0.105, GREY, scale=0.66, outline=True)
    taxi(x2, 0.105, GREY, scale=0.66, outline=True)
    ax.text(0.47, 0.052, "true current state for paired comparison only",
            ha="center", va="center", fontsize=9.4, color=GREY)
    ax.text(0.84, 0.105, "no action", ha="left", va="center", fontsize=9.2,
            color=GREY, fontweight="bold")

    # Operational action returns to SUMO; the clean twin remains read only.
    ax.add_patch(Circle((0.91, 0.27), 0.045, facecolor="#EEF6EF",
                        edgecolor=GREEN, linewidth=1.7, zorder=6))
    ax.text(0.91, 0.27, "GAT\npolicy", ha="center", va="center",
            fontsize=9.0, color=GREEN, fontweight="bold", zorder=7)
    arrow(ax, (0.82, 0.27), (0.865, 0.27), color=GREEN)
    ax.add_patch(FancyArrowPatch(
        (0.91, 0.315), (0.84, 0.43), arrowstyle="-|>",
        connectionstyle="arc3,rad=-0.25", mutation_scale=16,
        linewidth=1.7, color=GREEN,
    ))
    ax.text(0.925, 0.39, "action to\nSUMO", ha="center", va="center",
            fontsize=8.7, color=GREEN, fontweight="bold")

    ax.text(0.50, 0.005,
            "Tunnel entry is the trigger; stale telemetry is created at the observation boundary.",
            ha="center", va="bottom", fontsize=10.5, color="#202020",
            fontweight="bold")
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
