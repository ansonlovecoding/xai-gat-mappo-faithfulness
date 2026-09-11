#!/usr/bin/env python3
"""Create a reproducible, single-decision viva demonstration.

The script runs one frozen GAT checkpoint on held-out demand, waits for a
tunnel-triggered telemetry freeze, and captures one stale-exposed decision.
It writes a compact JSON record and a presentation-ready PNG. The decision
metrics are illustrative; the release verdict is read from the full offline
explanation audit and is never inferred from this one sample.

Usage:
  .venv/bin/python scripts/viva_demo.py
  .venv/bin/python scripts/viva_demo.py --guided --gui
  .venv/bin/python scripts/viva_demo.py --guided --auto-advance
  .venv/bin/python scripts/viva_demo.py --gui
  .venv/bin/python scripts/viva_demo.py --checkpoint path/to/ckpt_selected.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

import matplotlib

# Figures are saved to disk; SUMO owns the live GUI. Avoid opening a second
# native GUI backend when rendering the evidence in a terminal-only run.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sumolib
import torch
import traci
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# The presentation uses the TraCI GUI API directly. Keep the environment on
# that same connection even when libsumo happens to be installed.
os.environ["DISPATCH_MARL_FORCE_TRACI"] = "1"

from dispatch_marl import (  # noqa: E402
    DegradationConfig,
    DispatchEnv,
    DispatchEnvConfig,
    FaithfulnessConfig,
    FaithfulnessEvaluator,
    compute_attention_drift,
    seed_everything,
)
from dispatch_marl.env import AOI_MAX_S  # noqa: E402
from dispatch_marl.models import obs_dict_to_tensors  # noqa: E402
from dispatch_marl.runtime import load_policy  # noqa: E402
from dispatch_marl.scenario import demand_split_files  # noqa: E402

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "runs/dissertation_v10_corrected/training/B2_gat/seed_42/ckpt_selected.pt"
)
RELEASE_CHECKPOINT = (
    PROJECT_ROOT
    / "results/dissertation_v10_corrected/release/checkpoints/B2_gat_seed_42/ckpt_selected.pt"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "results/dissertation_v10_corrected/explanation_audit/audit_report.json"
)
DEFAULT_GUI_SETTINGS = PROJECT_ROOT / "configs/viva_demo.view.xml"
MACOS_SUMO_GUI_CANDIDATES = (
    Path.home() / ".local/opt/sumo-1.20-native/bin/sumo-gui",
    Path.home() / ".local/opt/sumo-1.20.0/bin/sumo-gui",
)
XQUARTZ_APP = Path("/Applications/Utilities/XQuartz.app")

COLORS = {
    "ink": "#17202A",
    "muted": "#5D6D7E",
    "clean": "#148F77",
    "stale": "#C0392B",
    "request": "#2874A6",
    "panel": "#F7F9FA",
    "line": "#CCD1D1",
    "warning": "#B9770E",
}


def _pause(args: argparse.Namespace, message: str = "Continue") -> None:
    if args.guided and not args.auto_advance:
        try:
            if input(f"\n[Enter] {message} | [q] finish: ").strip().lower() == "q":
                raise KeyboardInterrupt
        except EOFError:
            raise KeyboardInterrupt from None


def _stage(args: argparse.Namespace, number: int, title: str, text: str) -> None:
    if args.guided:
        print(f"\n{'=' * 72}\nSTEP {number}/6 — {title}\n{'=' * 72}\n{text}", flush=True)


def _present_results(record: dict[str, Any], args: argparse.Namespace) -> None:
    telemetry = record["telemetry"]
    explanation = record["explanation"]
    _stage(args, 4, "Paired telemetry and explanation outputs | 2 minutes",
           "The clean twin is a read-only observation from the SAME SUMO step.\n"
           "SUMO keeps moving; only the policy's telemetry is frozen.")
    print(f"Frozen position (m): {telemetry['observed_frozen_xy_m']}")
    print(f"True position (m):   {telemetry['sumo_true_xy_m']}")
    print(f"Position error: {telemetry['position_error_m']:.2f} m; "
          f"maximum graph AoI: {telemetry['max_aoi_s']:.2f} s")
    print("\nNode              Stale       Clean attention   Degraded attention")
    for label, valid, stale, clean, degraded in zip(
        explanation["node_labels"], explanation["node_mask"],
        explanation["stale_vehicle_mask"], explanation["clean_attention"],
        explanation["degraded_attention"],
    ):
        if valid:
            print(f"{label:<18}{str(bool(stale)):<12}{clean:>14.6f}{degraded:>21.6f}")
    print(f"\nWAMSN: {explanation['wamsn']:.6f} (age-weighted stale attention exposure)")
    print(f"{explanation['primary_def_name']}:")
    print(f"  clean={explanation['clean_twin_primary_def']:+.8f}; "
          f"degraded={explanation['primary_def']:+.8f}")
    print("DEF compares attention-ranked perturbations with type-matched random controls.\n"
          "For dispatch, the selected request is protected in the primary margin DEF.\n"
          "One decision illustrates the calculation; it cannot establish the study result.")
    print(f"\nOpen the generated figure: {args.output_dir / 'viva_demo.png'}")
    _pause(args, "Show the formal offline audit")
    audit = record["formal_audit"]
    _stage(args, 5, "Formal explanation-release audit | 1.5 minutes",
           f"Source: {args.audit_report.resolve()}\n"
           f"Model identifier: {args.model_id}\n"
           "This is a stored full-experiment verdict, not a verdict computed from this sample.")
    print(f"Decision: {audit['model_decision']}")
    for check in audit["failed_checks"]:
        print(f"  - {check}")
    print("WITHHOLD means keep attention as an internal diagnostic; it does not stop dispatch.\n"
          "Custom checkpoints are not automatically covered by this stored audit.")
    _pause(args, "Inspect saved evidence and finish")
    _stage(args, 6, "Evidence and implementation recap | 1 minute",
           f"JSON: {args.output_dir / 'viva_demo.json'}\n"
           f"Figure: {args.output_dir / 'viva_demo.png'}\n"
           "The JSON records the checkpoint, demand, model inputs/outputs, paired metrics,\n"
           "and offline audit summary. The figure presents the captured decision.\n"
           "Demonstrated: key functionality, model execution, outputs/results, system components.")


def _single_observation(
    obs: dict[str, torch.Tensor], index: int
) -> dict[str, torch.Tensor]:
    return {key: value[index : index + 1] for key, value in obs.items()}


def _as_tensors(
    obs: dict[str, np.ndarray], device: str
) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
        for key, value in obs.items()
    }


def _has_stale_vehicle(obs: dict[str, torch.Tensor]) -> bool:
    if float(obs["self"][0, 4].item()) > 0.0:
        return True
    taxi_aoi = obs["neighbor_taxis"][0, :, 4]
    taxi_mask = obs["neighbor_taxis_mask"][0].bool()
    return bool(((taxi_aoi > 0) & taxi_mask).any().item())


def _node_labels(k_neighbors: int, k_reservations: int) -> list[str]:
    return [
        "self",
        *(f"taxi {i}" for i in range(1, k_neighbors + 1)),
        *(f"request {i}" for i in range(1, k_reservations + 1)),
    ]


def _vehicle_positions(
    degraded: dict[str, torch.Tensor],
    clean: dict[str, torch.Tensor],
    node_index: int,
    bbox: tuple[float, float, float, float],
    net_diag: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed and true absolute positions for self or a taxi slot."""
    xmin, ymin, xmax, ymax = bbox
    width = max(1.0, xmax - xmin)
    height = max(1.0, ymax - ymin)

    def self_xy(item: dict[str, torch.Tensor]) -> np.ndarray:
        feature = item["self"][0]
        return np.array(
            [xmin + float(feature[0]) * width, ymin + float(feature[1]) * height]
        )

    observed_self = self_xy(degraded)
    true_self = self_xy(clean)
    if node_index == 0:
        return observed_self, true_self

    slot = node_index - 1
    observed_delta = degraded["neighbor_taxis"][0, slot, :2].cpu().numpy()
    true_delta = clean["neighbor_taxis"][0, slot, :2].cpu().numpy()
    return observed_self + observed_delta * net_diag, true_self + true_delta * net_diag


def _load_audit_summary(path: Path, model_id: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "overall_decision": "INCOMPLETE",
            "model_decision": "INCOMPLETE",
            "failed_checks": ["formal audit report not found"],
        }
    report = json.loads(path.read_text(encoding="utf-8"))
    model = next(
        (item for item in report.get("models", []) if item.get("model_id") == model_id),
        None,
    )
    if model is None:
        return {
            "overall_decision": report.get("overall_decision", "INCOMPLETE"),
            "model_decision": "INCOMPLETE",
            "failed_checks": [f"model {model_id} not found in audit report"],
        }
    failed = [
        check["name"]
        for check in model.get("checks", [])
        if check.get("status") in {"FAIL", "INDETERMINATE", "INCOMPLETE"}
    ]
    return {
        "overall_decision": report.get("overall_decision", "INCOMPLETE"),
        "model_decision": model.get("decision", "INCOMPLETE"),
        "failed_checks": failed,
    }


def _primary_def(result: Any) -> tuple[float, str]:
    if result.excl_evaluated and math.isfinite(result.def_m_excl):
        return float(result.def_m_excl), "action-protected margin DEF"
    return float(result.def_margin), "margin DEF"


def _missing_macos_libraries(binary: Path) -> list[str]:
    """Return absolute Homebrew libraries missing from a Mach-O binary."""
    if platform.system() != "Darwin":
        return []
    try:
        output = subprocess.run(
            ["otool", "-L", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    libraries = [
        Path(line.strip().split(" (", 1)[0])
        for line in output.splitlines()[1:]
        if line.strip().startswith(("/usr/local/", "/opt/homebrew/"))
    ]
    return [str(path) for path in libraries if not path.exists()]


def _prepare_gui(args: argparse.Namespace) -> None:
    """Select a compatible GUI binary and start the macOS X11 server."""
    if not args.gui:
        return
    if platform.system() == "Darwin":
        compatible_gui = next(
            (path for path in MACOS_SUMO_GUI_CANDIDATES if path.exists()), None
        )
        if compatible_gui is not None:
            os.environ["GUISIM_BINARY"] = str(compatible_gui)
    gui_binary = Path(sumolib.checkBinary("sumo-gui"))
    missing = _missing_macos_libraries(gui_binary)
    if missing:
        details = "\n".join(f"  - {library}" for library in missing)
        raise RuntimeError(
            "SUMO GUI cannot start because native libraries are missing:\n"
            f"{details}\nRun without --gui or install a compatible SUMO GUI."
        )
    if platform.system() != "Darwin":
        return
    if not XQUARTZ_APP.exists():
        raise RuntimeError(
            "SUMO GUI on macOS requires XQuartz. Install it in a terminal with:\n"
            "  brew install --cask xquartz\n"
            "Then log out and back in once, or start XQuartz manually."
        )
    subprocess.run(["open", "-a", "XQuartz"], check=True)
    socket = Path("/tmp/.X11-unix/X0")
    for _ in range(40):
        if socket.exists():
            break
        time.sleep(0.25)
    if not os.environ.get("DISPLAY"):
        display = subprocess.run(
            ["launchctl", "getenv", "DISPLAY"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        os.environ["DISPLAY"] = display or ":0"
    glxinfo = Path("/opt/X11/bin/glxinfo")
    if glxinfo.exists():
        glx = subprocess.run(
            [str(glxinfo), "-B"],
            capture_output=True,
            text=True,
            env=os.environ,
        )
        if glx.returncode != 0:
            diagnostic = "\n".join(
                line
                for line in (glx.stderr + glx.stdout).splitlines()
                if "error:" in line.lower() or "X Error" in line
            )
            raise RuntimeError(
                "XQuartz cannot create an OpenGL context on this Mac. "
                "SUMO GUI cannot render until GLX works.\n"
                f"{diagnostic or 'glxinfo failed without a diagnostic message.'}"
            )
    print(f"SUMO GUI:   {gui_binary}")
    print(f"X11 display: {os.environ['DISPLAY']}")


def _prepare_tunnel_overlay(env: DispatchEnv, args: argparse.Namespace) -> None:
    """Add a persistent, high-contrast overlay for configured tunnel edges."""
    if not args.gui:
        return
    try:
        views = traci.gui.getIDList()
        if not views:
            return
        for vehicle_id in traci.vehicle.getIDList():
            traci.vehicle.setColor(vehicle_id, (150, 158, 166, 255))
        tunnel_shapes: list[list[tuple[float, float]]] = []
        for edge_id in sorted(env.scenario.tunnel_edges):
            lane_count = traci.edge.getLaneNumber(edge_id)
            if lane_count < 1:
                continue
            shape = list(traci.lane.getShape(f"{edge_id}_0"))
            if len(shape) < 2:
                continue
            tunnel_shapes.append(shape)
            for suffix, color, width, layer in (
                ("outline", (205, 85, 20, 255), 18, 90),
                ("center", (255, 195, 35, 255), 7, 91),
            ):
                polygon_id = f"VIVA_TUNNEL_{edge_id}_{suffix}"
                if polygon_id not in set(traci.polygon.getIDList()):
                    traci.polygon.add(
                        polygon_id,
                        shape,
                        color,
                        fill=False,
                        polygonType="telemetry-loss tunnel",
                        layer=layer,
                        lineWidth=width,
                    )

        if not tunnel_shapes:
            return
        longest = max(tunnel_shapes, key=len)
        tunnel_label_xy = longest[len(longest) // 2]
        _add_demo_poi(
            "TUNNEL_ZONE",
            tunnel_label_xy,
            (230, 135, 20, 255),
            "TELEMETRY-LOSS TUNNEL",
        )
    except traci.exceptions.TraCIException as exc:
        print(f"warning: could not prepare tunnel view: {exc}", flush=True)


def _request_candidates(
    agents: list[str],
    logits: torch.Tensor,
    batched: dict[str, torch.Tensor],
    env: DispatchEnv,
    request_id: str | None,
    selected_taxi: str,
) -> list[dict[str, Any]]:
    """Return taxis whose current local action space contains one request."""
    if request_id is None:
        return []
    probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    reservation = next(
        (
            item
            for item in traci.person.getTaxiReservations(0)
            if item.id == request_id
        ),
        None,
    )
    pickup_xy = (
        traci.simulation.convert2D(reservation.fromEdge, 0.0, laneIndex=0)
        if reservation is not None
        else None
    )
    candidates: list[dict[str, Any]] = []
    for index, taxi_id in enumerate(agents):
        visible_requests = list(env.reservation_ids(taxi_id))
        if request_id not in visible_requests:
            continue
        request_action = visible_requests.index(request_id) + 1
        taxi_xy = traci.vehicle.getPosition(taxi_id)
        candidates.append(
            {
                "taxi_id": taxi_id,
                "request_probability": float(probabilities[index, request_action]),
                "distance_to_pickup_m": (
                    float(math.dist(taxi_xy, pickup_xy))
                    if pickup_xy is not None
                    else math.inf
                ),
                "stale_self_telemetry": bool(
                    batched["position_valid"][index].detach().cpu().item() < 0.5
                ),
            }
        )
    candidates.sort(key=lambda item: item["distance_to_pickup_m"])
    shown = candidates[:5]
    if not any(item["taxi_id"] == selected_taxi for item in shown):
        selected = next(
            (item for item in candidates if item["taxi_id"] == selected_taxi), None
        )
        if selected is not None:
            shown[-1:] = [selected]
    return shown


def _capture_decision(args: argparse.Namespace) -> tuple[dict[str, Any], DispatchEnv]:
    device = args.device
    seed_everything(args.eval_seed, deterministic_torch=True)
    policy, checkpoint = load_policy(args.checkpoint, device)
    if checkpoint.get("policy_type", "gat") != "gat":
        raise ValueError("the viva demo requires a GAT checkpoint with attention weights")

    env_config = checkpoint.get("env_config", {})
    area = str(env_config.get("area", "central_park"))
    demand_files = demand_split_files(area, "test")
    if args.demand_file:
        demand_files = [args.demand_file]

    _stage(args, 2, "Load the trained model | 1 minute",
           f"Checkpoint: {args.checkpoint.resolve()}\n"
           f"Epoch: {checkpoint.get('epoch')}; device: {device}\n"
           f"Parameters: {sum(p.numel() for p in policy.parameters()):,}\n"
           f"Policy configuration: {policy.config}\n"
           f"Demand: {demand_files[0]}\n"
           "Execution: policy.eval() + no_grad(); categorical sampling from action logits.\n"
           "Each taxi has its own action distribution; candidate probabilities are not\n"
           "a single probability distribution over competing taxis.")
    _pause(args, "Run inference and capture a stale-exposed decision")
    if args.guided:
        print("Running SUMO and the loaded policy; searching for a stale-exposed decision ...", flush=True)

    env = DispatchEnv(
        DispatchEnvConfig(
            area=area,
            n_taxis=int(env_config.get("n_taxis", 20)),
            k_neighbors=policy.config.k_neighbors,
            k_reservations=policy.config.k_reservations,
            step_length_s=int(env_config.get("step_length_s", 10)),
            seed=args.eval_seed,
            degradation=DegradationConfig(
                mode="tunnel_triggered",
                corruption="freeze",
                outage_duration_s=args.outage_duration,
            ),
            emit_clean_obs=True,
            use_gui=args.gui,
            gui_settings_file=str(DEFAULT_GUI_SETTINGS) if args.gui else None,
            gui_delay_ms=args.gui_step_delay_ms,
            gui_simulation_step_s=args.gui_simulation_step_s if args.gui else None,
            gui_window_size=tuple(args.gui_window_size) if args.gui else None,
            gui_window_position=(0, 20) if args.gui else None,
        )
    )
    evaluator = FaithfulnessEvaluator(
        policy,
        FaithfulnessConfig(
            top_k_values=(1, 2, 3),
            n_random_baselines=5,
            seed=args.eval_seed,
            exclusion_variant=True,
            random_baseline="type_matched",
        ),
    )

    fallback: tuple[Any, ...] | None = None
    dispatch_fallback: tuple[Any, ...] | None = None
    try:
        for episode, demand_file in enumerate(demand_files[: args.max_episodes]):
            episode_seed = args.eval_seed + episode
            seed_everything(episode_seed, deterministic_torch=True)
            obs_dict, _ = env.reset(
                seed=episode_seed, options={"taxi_route_file": demand_file}
            )
            _prepare_tunnel_overlay(env, args)
            step = 0
            while not env.done and step < args.max_steps:
                step += 1
                if not obs_dict:
                    obs_dict, *_ = env.step({})
                    continue
                batched, agents = obs_dict_to_tensors(obs_dict, device=device)
                with torch.no_grad():
                    logits = policy.forward(batched)["logits"]
                    actions_tensor = torch.distributions.Categorical(logits=logits).sample()
                actions_np = actions_tensor.cpu().numpy().astype(int)
                actions = {agent: int(actions_np[i]) for i, agent in enumerate(agents)}

                for index, agent in enumerate(agents):
                    degraded = _single_observation(batched, index)
                    clean_obs = env.last_clean_obs.get(agent)
                    if clean_obs is None or not _has_stale_vehicle(degraded):
                        continue
                    action = int(actions_np[index])
                    visible_requests = env.reservation_ids(agent)
                    request_id = (
                        visible_requests[action - 1]
                        if action > 0 and action - 1 < len(visible_requests)
                        else None
                    )
                    request_candidates = _request_candidates(
                        agents, logits, batched, env, request_id, agent
                    )
                    candidate = (
                        episode,
                        demand_file,
                        step,
                        float(env.sim_time),
                        env.network_bounds,
                        env.network_diagonal_m,
                        agent,
                        action,
                        degraded,
                        _as_tensors(clean_obs, device),
                        env.neighbor_ids(agent),
                        env.reservation_ids(agent),
                        request_candidates,
                    )
                    if fallback is None:
                        fallback = candidate
                    if action >= 1 and dispatch_fallback is None:
                        dispatch_fallback = candidate
                    if action >= 1 and any(
                        item["stale_self_telemetry"]
                        and item["distance_to_pickup_m"] <= 800.0
                        for item in request_candidates
                    ):
                        record = _score_candidate(
                            candidate, evaluator, checkpoint, args, device
                        )
                        record["scope"]["live_scene_matches_capture"] = True
                        return record, env
                obs_dict, *_ = env.step(actions)
        selected_fallback = dispatch_fallback or fallback
        if selected_fallback is not None:
            record = _score_candidate(
                selected_fallback, evaluator, checkpoint, args, device
            )
            record["scope"]["live_scene_matches_capture"] = False
            return record, env
    except BaseException:
        env.close()
        raise

    env.close()
    raise RuntimeError(
        "no stale-exposed decision was found; increase --max-steps or "
        "--max-episodes, or verify the scenario tunnel manifest"
    )


def _advance_gui_for(
    duration_s: float, on_step: Callable[[], bool] | None = None
) -> bool:
    """Advance SUMO for a wall-clock duration so movement remains visible."""
    deadline = time.monotonic() + max(0.0, duration_s)
    while time.monotonic() < deadline:
        try:
            if traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                if on_step is not None and on_step():
                    return True
            else:
                time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException):
            return True
    return False


def _run_gui_presentation(record: dict[str, Any], args: argparse.Namespace) -> None:
    started = time.monotonic()
    context = _annotate_gui(record, args)
    remaining = (args.journey_seconds if args.guided else
                 args.gui_duration_seconds - (time.monotonic() - started))
    if remaining > 0:
        print(
            f"Phase 5: live pickup and drop-off for about "
            f"{remaining / 60:.1f} minutes",
            flush=True,
        )
        _advance_gui_for(remaining, _pickup_tracker(context, args))
    if args.close_gui_on_finish or args.guided:
        return
    print(
        "Timed presentation complete. SUMO GUI will remain open; "
        "close the GUI window or press Ctrl+C to finish.",
        flush=True,
    )
    while True:
        try:
            if not traci.gui.getIDList():
                return
        except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException):
            return
        time.sleep(0.5)


def _pickup_tracker(
    context: dict[str, Any] | None, args: argparse.Namespace
) -> Callable[[], bool] | None:
    """Follow one dispatch through pickup and drop-off, then freeze the view."""
    if context is None or context.get("assigned_taxi") is None:
        return None
    view = context["view"]
    assigned_taxi = context["assigned_taxi"]
    state = {
        "mode": "assigned",
        "pickup_seen": False,
        "dropoff_seen": False,
        "frame": 0,
        "styled_vehicle_ids": set(),
    }

    def update() -> bool:
        vehicle_ids = set(traci.vehicle.getIDList())
        if assigned_taxi not in vehicle_ids:
            return state["pickup_seen"]
        new_vehicle_ids = vehicle_ids - state["styled_vehicle_ids"]
        for vehicle_id in new_vehicle_ids:
            traci.vehicle.setColor(
                vehicle_id,
                (20, 170, 90, 255)
                if vehicle_id == assigned_taxi
                else (150, 158, 166, 255),
            )
        state["styled_vehicle_ids"].update(new_vehicle_ids)
        state["frame"] += 1
        position = traci.vehicle.getPosition(assigned_taxi)
        color = (20, 170, 90, 255)
        occupied = assigned_taxi in set(traci.vehicle.getTaxiFleet(2))
        idle = assigned_taxi in set(traci.vehicle.getTaxiFleet(0))
        if state["pickup_seen"] and idle:
            if not state["dropoff_seen"]:
                state["dropoff_seen"] = True
                for poi_id in ("PICKUP_COMPLETE", "ORDER_ASSIGNED"):
                    if poi_id in set(traci.poi.getIDList()):
                        traci.poi.remove(poi_id)
                _add_demo_poi(
                    "DROPOFF_COMPLETE",
                    position,
                    color,
                    f"PASSENGER DROPPED OFF: {assigned_taxi}",
                )
                print(
                    f"Drop-off reached: journey completed by {assigned_taxi}",
                    flush=True,
                )
                traci.gui.screenshot(
                    view,
                    str(args.output_dir / "sumo_dropoff_phase.png"),
                )
                traci.simulationStep()
            return True
        if occupied:
            if not state["pickup_seen"]:
                state["pickup_seen"] = True
                state["mode"] = "occupied"
                for poi_id in ("ORDER_ASSIGNED", "PASSENGER_REQUEST"):
                    if poi_id in set(traci.poi.getIDList()):
                        traci.poi.remove(poi_id)
                _add_demo_poi(
                    "PICKUP_COMPLETE",
                    position,
                    color,
                    f"PASSENGER PICKED UP: {assigned_taxi}",
                )
                print(
                    f"Pickup reached: passenger entered {assigned_taxi}",
                    flush=True,
                )
                traci.gui.screenshot(
                    view,
                    str(args.output_dir / "sumo_pickup_phase.png"),
                )
            elif state["frame"] % 4 == 0:
                traci.poi.setPosition(
                    "PICKUP_COMPLETE", float(position[0]), float(position[1])
                )
            return False
        waiting = traci.vehicle.getSpeed(assigned_taxi) < 0.2
        mode = "waiting" if waiting else "moving"
        if mode != state["mode"]:
            previous_mode = state["mode"]
            state["mode"] = mode
            print(
                "Assigned taxi is waiting in traffic."
                if waiting
                else (
                    "Assigned taxi is travelling to the passenger."
                    if previous_mode == "assigned"
                    else "Assigned taxi is moving again."
                ),
                flush=True,
            )
            _add_demo_poi(
                "ORDER_ASSIGNED",
                position,
                color,
                f"WAITING IN TRAFFIC: {assigned_taxi}"
                if waiting
                else f"TO PICKUP: {assigned_taxi}",
            )
        elif state["frame"] % 4 == 0:
            traci.poi.setPosition(
                "ORDER_ASSIGNED", float(position[0]), float(position[1])
            )
        return False

    return update


def _annotate_gui(
    record: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any] | None:
    """Present request, candidates, stale evidence, and assignment in order."""
    if not args.gui:
        return
    try:
        views = traci.gui.getIDList()
        if not views:
            return None
        view = views[0]
        assigned_taxi = record["scope"]["agent"]
        stale_taxi = record["telemetry"].get("focus_vehicle_id")
        request_id = record["decision"].get("reservation_id")
        candidates = record["decision"].get("request_candidates", [])
        vehicle_ids = set(traci.vehicle.getIDList())
        pickup_xy: tuple[float, float] | None = None
        dropoff_xy: tuple[float, float] | None = None

        for vehicle_id in vehicle_ids:
            traci.vehicle.setColor(vehicle_id, (150, 158, 166, 255))

        live_reservations = {
            reservation.id: reservation
            for reservation in traci.person.getTaxiReservations(0)
        }
        reservation = live_reservations.get(request_id)
        if reservation is not None:
            pickup_xy = traci.simulation.convert2D(
                reservation.fromEdge, 0.0, laneIndex=0
            )
            dropoff_xy = traci.simulation.convert2D(
                reservation.toEdge, 0.0, laneIndex=0
            )

        def show_phase(
            number: int,
            description: str,
            points: list[tuple[float, float]],
            filename: str,
        ) -> None:
            if not points:
                return
            traci.gui.trackVehicle(view, "")
            _focus_gui(
                view,
                points,
                transition_seconds=args.gui_transition_seconds,
            )
            traci.gui.screenshot(view, str(args.output_dir / filename))
            traci.simulationStep()
            timing = ("automatic rehearsal" if args.auto_advance else "press Enter in terminal") if args.guided else f"{args.gui_phase_seconds:g} seconds"
            print(
                f"Phase {number}: {description} ({timing})",
                flush=True,
            )
            if args.guided:
                _pause(args, "Next SUMO view (switch back to this terminal)")
            else:
                time.sleep(args.gui_phase_seconds)

        request_points = [pickup_xy] if pickup_xy is not None else []
        tunnel_shapes = [
            list(traci.polygon.getShape(polygon_id))
            for polygon_id in traci.polygon.getIDList()
            if polygon_id.startswith("VIVA_TUNNEL_")
            and polygon_id.endswith("_center")
        ]
        if pickup_xy is not None and tunnel_shapes:
            nearest_tunnel = min(
                tunnel_shapes,
                key=lambda shape: min(math.dist(pickup_xy, point) for point in shape),
            )
            request_points.extend(nearest_tunnel)
        if pickup_xy is not None:
            _add_demo_poi(
                "PASSENGER_REQUEST",
                pickup_xy,
                (35, 105, 190, 255),
                f"NEW REQUEST: {request_id}",
            )
        if dropoff_xy is not None:
            _add_demo_poi(
                "PASSENGER_DROPOFF",
                dropoff_xy,
                (125, 75, 170, 255),
                "DROPOFF DESTINATION",
            )
        show_phase(
            1,
            "passenger request beside the highlighted tunnel",
            request_points,
            "sumo_request_phase.png",
        )

        candidate_points = list(request_points)
        print("\nCandidate taxis that can see this request", flush=True)
        for rank, candidate in enumerate(candidates, start=1):
            taxi_id = candidate["taxi_id"]
            if taxi_id not in set(traci.vehicle.getIDList()):
                continue
            position = traci.vehicle.getPosition(taxi_id)
            candidate_points.append(position)
            probability = candidate["request_probability"]
            distance = candidate["distance_to_pickup_m"]
            color = (210, 155, 25, 255)
            traci.vehicle.setColor(taxi_id, color)
            _add_demo_poi(
                f"CANDIDATE_{rank}",
                position,
                color,
                f"C{rank} {taxi_id}",
                highlight=False,
            )
            print(
                f"  C{rank} {taxi_id}: {distance:.0f} m away, "
                f"P(request)={probability:.3f}",
                flush=True,
            )
        show_phase(
            2,
            "nearby candidate taxis and request probabilities",
            candidate_points,
            "sumo_candidates_phase.png",
        )

        for rank in range(1, len(candidates) + 1):
            poi_id = f"CANDIDATE_{rank}"
            if poi_id in set(traci.poi.getIDList()):
                traci.poi.remove(poi_id)
        stale_points = list(request_points)
        frozen_xy = tuple(record["telemetry"]["observed_frozen_xy_m"])
        stale_points.append(frozen_xy)
        if stale_taxi in set(traci.vehicle.getIDList()):
            stale_xy = traci.vehicle.getPosition(stale_taxi)
            stale_points.append(stale_xy)
            traci.vehicle.setColor(stale_taxi, (205, 45, 45, 255))
            _add_demo_poi(
                "STALE_TAXI",
                stale_xy,
                (205, 45, 45, 255),
                f"STALE TELEMETRY: {stale_taxi}",
            )
        _add_demo_poi(
            "TUNNEL_SIGNAL_LOSS",
            frozen_xy,
            (230, 135, 20, 255),
            "LAST VALID POSITION",
        )
        show_phase(
            3,
            "stale taxi and its tunnel-triggered frozen position",
            stale_points,
            "sumo_tunnel_phase.png",
        )

        for poi_id in ("STALE_TAXI", "TUNNEL_SIGNAL_LOSS"):
            if poi_id in set(traci.poi.getIDList()):
                traci.poi.remove(poi_id)
        vehicle_ids = set(traci.vehicle.getIDList())
        assigned_xy = (
            traci.vehicle.getPosition(assigned_taxi)
            if assigned_taxi in vehicle_ids
            else None
        )
        assignment_points = [
            point for point in (assigned_xy, pickup_xy) if point is not None
        ]
        if assigned_xy is not None:
            traci.vehicle.setColor(assigned_taxi, (20, 170, 90, 255))
            traci.vehicle.highlight(
                assigned_taxi,
                color=(20, 170, 90, 255),
                size=22,
                alphaMax=255,
                duration=3600,
            )
            _add_demo_poi(
                "ORDER_ASSIGNED",
                assigned_xy,
                (20, 170, 90, 255),
                f"POLICY SELECTED: {assigned_taxi}",
            )
        if pickup_xy is not None:
            _add_demo_poi(
                "PASSENGER_REQUEST",
                pickup_xy,
                (35, 105, 190, 255),
                f"PICKUP: request {request_id}",
            )
        if request_id is not None and assigned_taxi in set(
            traci.vehicle.getTaxiFleet(0)
        ):
            traci.vehicle.dispatchTaxi(assigned_taxi, [request_id])
        show_phase(
            4,
            "sampled policy assignment",
            assignment_points,
            "sumo_order_phase.png",
        )
        if assigned_taxi in vehicle_ids:
            _focus_gui(
                view,
                [traci.vehicle.getPosition(assigned_taxi)],
                transition_seconds=args.gui_transition_seconds,
            )
            traci.gui.trackVehicle(view, assigned_taxi)
            traci.gui.setZoom(view, 1200)
        return {
            "view": view,
            "assigned_taxi": assigned_taxi if assigned_taxi in vehicle_ids else None,
            "request_id": request_id,
            "pickup_xy": pickup_xy,
            "dropoff_xy": dropoff_xy,
        }
    except traci.exceptions.TraCIException as exc:
        print(f"warning: could not add all SUMO GUI annotations: {exc}", flush=True)
        return None


def _add_demo_poi(
    poi_id: str,
    xy: tuple[float, float],
    color: tuple[int, int, int, int],
    label: str,
    *,
    highlight: bool = True,
) -> None:
    existing = set(traci.poi.getIDList())
    if poi_id not in existing:
        traci.poi.add(
            poi_id,
            float(xy[0]),
            float(xy[1]),
            color,
            poiType=label,
            layer=100,
            width=12,
            height=12,
        )
    else:
        traci.poi.setPosition(poi_id, float(xy[0]), float(xy[1]))
    traci.poi.setParameter(poi_id, "PARAM_TEXT", label)
    if highlight:
        traci.poi.highlight(
            poi_id, color=color, size=28, alphaMax=255, duration=3600
        )


def _focus_gui(
    view: str,
    points: list[tuple[float, float]],
    *,
    transition_seconds: float = 0.0,
) -> None:
    xs, ys = zip(*points)
    span = max(max(xs) - min(xs), max(ys) - min(ys), 180.0)
    margin = max(60.0, span * 0.35)
    target = (
        min(xs) - margin,
        min(ys) - margin,
        max(xs) + margin,
        max(ys) + margin,
    )
    if transition_seconds > 0:
        (x0, y0), (x1, y1) = traci.gui.getBoundary(view)
        current = (x0, y0, x1, y1)
        frames = max(2, round(transition_seconds * 30))
        frame_delay = transition_seconds / frames
        for frame in range(1, frames + 1):
            progress = frame / frames
            eased = progress * progress * (3.0 - 2.0 * progress)
            boundary = tuple(
                start + (end - start) * eased
                for start, end in zip(current, target)
            )
            traci.gui.setBoundary(view, *boundary)
            time.sleep(frame_delay)
    else:
        traci.gui.setBoundary(view, *target)
    # GUI commands are applied immediately but FOX redraws the canvas on the
    # next simulation event. This post-capture step is presentation-only and
    # does not alter the recorded decision or audit metrics.
    traci.simulationStep()


def _score_candidate(
    candidate: tuple[Any, ...],
    evaluator: FaithfulnessEvaluator,
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    (
        episode,
        demand_file,
        step,
        sim_time,
        network_bounds,
        network_diagonal_m,
        agent,
        action,
        degraded,
        clean,
        neighbor_ids,
        reservation_ids,
        request_candidates,
    ) = candidate
    degraded_result, clean_result = evaluator.evaluate_paired_decision(
        degraded, clean, action=action
    )
    drift = compute_attention_drift(
        clean_result.attention_row, degraded_result.attention_row
    )
    stale_indices = np.flatnonzero(degraded_result.stale_vehicle_mask)
    focus_node = int(
        stale_indices[np.argmax(degraded_result.attention_row[stale_indices])]
    )
    observed_xy, true_xy = _vehicle_positions(
        degraded,
        clean,
        focus_node,
        network_bounds,
        network_diagonal_m,
    )
    primary_def, primary_def_name = _primary_def(degraded_result)
    clean_primary_def, _ = _primary_def(clean_result)
    labels = _node_labels(
        evaluator.policy.config.k_neighbors,
        evaluator.policy.config.k_reservations,
    )
    audit = _load_audit_summary(args.audit_report, args.model_id)
    focus_vehicle_id = (
        agent
        if focus_node == 0
        else neighbor_ids[focus_node - 1]
        if focus_node - 1 < len(neighbor_ids)
        else None
    )
    reservation_id = (
        reservation_ids[action - 1]
        if action > 0 and action - 1 < len(reservation_ids)
        else None
    )
    with torch.no_grad():
        forward_output = evaluator.policy.forward(degraded)
        action_probabilities = torch.softmax(forward_output["logits"], dim=-1)[0]
    return {
        "model_execution": {
            "input_shapes": {key: list(value.shape) for key, value in degraded.items()},
            "output_shapes": {
                key: list(value.shape) for key, value in forward_output.items()
                if isinstance(value, torch.Tensor)
            },
            "action_probabilities": action_probabilities.detach().cpu().tolist(),
            "reservation_ids": list(reservation_ids),
            "action_rule": "categorical sampling (not argmax)",
        },
        "scope": {
            "kind": "illustrative single decision",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "training_seed": checkpoint.get("seed", args.training_seed),
            "evaluation_seed": args.eval_seed,
            "episode": int(episode),
            "demand_file": demand_file,
            "step": int(step),
            "sim_time_s": sim_time,
            "agent": agent,
            "device": device,
            "condition": f"tunnel-triggered {args.outage_duration:g}s freeze",
        },
        "decision": {
            "action_index": int(action),
            "action_label": "no-op" if action == 0 else f"dispatch request {action}",
            "reservation_id": reservation_id,
            "request_candidates": request_candidates,
            "probability": float(degraded_result.pi_full),
        },
        "telemetry": {
            "focus_node_index": focus_node,
            "focus_node_label": labels[focus_node],
            "focus_vehicle_id": focus_vehicle_id,
            "observed_frozen_xy_m": observed_xy.tolist(),
            "sumo_true_xy_m": true_xy.tolist(),
            "position_error_m": float(np.linalg.norm(true_xy - observed_xy)),
            "max_aoi_s": float(degraded_result.max_aoi_s),
            "stale_vehicle_count": int(degraded_result.n_stale_vehicle),
        },
        "explanation": {
            "node_labels": labels,
            "node_mask": degraded_result.node_mask.tolist(),
            "stale_vehicle_mask": degraded_result.stale_vehicle_mask.tolist(),
            "degraded_attention": degraded_result.attention_row.tolist(),
            "clean_attention": clean_result.attention_row.tolist(),
            "wamsn": float(degraded_result.wamsn),
            "stale_attention_share": float(degraded_result.stale_attention_share),
            "attention_drift_js": float(drift),
            "probability_def": float(degraded_result.def_score),
            "margin_def": float(degraded_result.def_margin),
            "action_protected_margin_def": (
                float(degraded_result.def_m_excl)
                if degraded_result.excl_evaluated
                else None
            ),
            "primary_def_name": primary_def_name,
            "primary_def": primary_def,
            "clean_twin_primary_def": clean_primary_def,
            "primary_def_change": primary_def - clean_primary_def,
        },
        "formal_audit": audit,
        "note": (
            "Decision metrics illustrate the pipeline only. The formal audit "
            "decision comes from the complete held-out multi-checkpoint evidence."
        ),
    }


def _rounded_panel(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor=COLORS["panel"],
            edgecolor=COLORS["line"],
            linewidth=1.0,
            clip_on=False,
            zorder=-10,
        )
    )


def _render_figure(record: dict[str, Any], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
        }
    )
    fig = plt.figure(figsize=(14, 8), facecolor="white")
    grid = fig.add_gridspec(
        2, 3, height_ratios=[0.18, 0.82], width_ratios=[1.05, 1.45, 1.05],
        hspace=0.12, wspace=0.22,
    )
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(
        0,
        0.72,
        "Viva demo: one stale-exposed dispatch decision",
        fontsize=21,
        fontweight="bold",
        color=COLORS["ink"],
    )
    scope = record["scope"]
    title_ax.text(
        0,
        0.25,
        f"GAT checkpoint | {scope['condition'].replace('60s', '60 s')} | {scope['demand_file']} | "
        f"t = {scope['sim_time_s']:.0f}s",
        fontsize=11,
        color=COLORS["muted"],
    )

    state_ax = fig.add_subplot(grid[1, 0])
    _rounded_panel(state_ax)
    telemetry = record["telemetry"]
    observed = np.asarray(telemetry["observed_frozen_xy_m"])
    truth = np.asarray(telemetry["sumo_true_xy_m"])
    offset = truth - observed
    scale = max(20.0, float(np.max(np.abs(offset))) * 1.7)
    state_ax.scatter(
        0, 0, s=170, color=COLORS["stale"], marker="X", zorder=3,
        label="frozen observation",
    )
    state_ax.scatter(
        offset[0], offset[1], s=170, color=COLORS["clean"], marker="o", zorder=3,
        label="SUMO truth / clean twin",
    )
    state_ax.annotate(
        "",
        xy=offset,
        xytext=(0, 0),
        arrowprops={"arrowstyle": "->", "color": COLORS["ink"], "lw": 1.8},
    )
    state_ax.set_xlim(min(-scale, offset[0] - scale * 0.35), max(scale, offset[0] + scale * 0.35))
    state_ax.set_ylim(min(-scale, offset[1] - scale * 0.35), max(scale, offset[1] + scale * 0.35))
    state_ax.set_aspect("equal", adjustable="box")
    state_ax.grid(color="#E5E8E8", linewidth=0.7)
    state_ax.set_xlabel("east-west displacement (m)")
    state_ax.set_ylabel("north-south displacement (m)")
    state_ax.set_title("1. Telemetry state", loc="left", color=COLORS["ink"])
    state_ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    state_ax.text(
        0.03,
        0.04,
        f"Node: {telemetry['focus_node_label']}\n"
        f"AoI: {telemetry['max_aoi_s']:.0f} s\n"
        f"position error: {telemetry['position_error_m']:.1f} m",
        transform=state_ax.transAxes,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": COLORS["line"], "pad": 6},
    )

    attention_ax = fig.add_subplot(grid[1, 1])
    _rounded_panel(attention_ax)
    explanation = record["explanation"]
    labels = np.asarray(explanation["node_labels"])
    valid = np.asarray(explanation["node_mask"], dtype=bool)
    stale = np.asarray(explanation["stale_vehicle_mask"], dtype=bool)
    degraded_attention = np.asarray(explanation["degraded_attention"])[valid]
    clean_attention = np.asarray(explanation["clean_attention"])[valid]
    shown_labels = labels[valid]
    shown_stale = stale[valid]
    y = np.arange(len(shown_labels))
    colors = [COLORS["stale"] if item else COLORS["request"] for item in shown_stale]
    attention_ax.barh(
        y, degraded_attention, color=colors, alpha=0.88, label="degraded observation"
    )
    attention_ax.scatter(
        clean_attention, y, s=42, facecolor="white", edgecolor=COLORS["clean"],
        linewidth=1.6, zorder=3, label="clean twin",
    )
    attention_ax.set_yticks(y, shown_labels)
    attention_ax.invert_yaxis()
    attention_ax.set_xlabel("aggregated self-row attention")
    attention_ax.set_title("2. Explanation channel", loc="left", color=COLORS["ink"])
    attention_ax.grid(axis="x", color="#E5E8E8", linewidth=0.7)
    attention_ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.065),
        ncol=2,
        fontsize=9,
    )
    attention_ax.text(
        0.98,
        0.98,
        "red = stale vehicle node",
        transform=attention_ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=COLORS["stale"],
    )

    result_ax = fig.add_subplot(grid[1, 2])
    _rounded_panel(result_ax)
    result_ax.axis("off")
    result_ax.set_title("3. Audit interpretation", loc="left", color=COLORS["ink"], pad=12)
    decision = record["decision"]
    audit = record["formal_audit"]
    def line(y: float, text: str, color: str, size: float, weight: str = "normal") -> None:
        result_ax.text(
            0.06, y, text, transform=result_ax.transAxes, color=color,
            fontsize=size, fontweight=weight, va="top",
        )

    line(0.91, "POLICY DECISION", COLORS["muted"], 9, "bold")
    display_action = ("no-op" if decision["action_index"] == 0 else
                      f"dispatch slot {decision['action_index']}")
    line(0.83, display_action, COLORS["ink"], 17, "bold")
    line(0.78, f"reservation ID: {decision['reservation_id']}", COLORS["muted"], 9)
    line(0.73, f"chosen-action probability  {decision['probability']:.3f}", COLORS["muted"], 10)
    line(0.65, "ILLUSTRATIVE METRICS", COLORS["muted"], 9, "bold")
    line(0.58, f"WAMSN  {explanation['wamsn']:.3f}", COLORS["ink"], 13, "bold")
    line(0.51, f"stale attention share  {explanation['stale_attention_share']:.3f}", COLORS["ink"], 10.5)
    line(0.45, f"attention drift (JS)  {explanation['attention_drift_js']:.6f}", COLORS["ink"], 9)
    line(0.39, explanation['primary_def_name'], COLORS["ink"], 9)
    line(0.35, f"DEF {explanation['primary_def']:+.8f}", COLORS["ink"], 10)
    line(0.30, f"Paired change {explanation['primary_def_change']:+.8f}", COLORS["ink"], 9)
    line(0.23, "FORMAL OFFLINE AUDIT", COLORS["muted"], 9, "bold")
    verdict_color = COLORS["stale"] if audit["model_decision"] == "WITHHOLD" else COLORS["clean"]
    line(0.16, audit["model_decision"], verdict_color, 22, "bold")
    failed = audit.get("failed_checks", [])
    if failed:
        line(0.075, f"{len(failed)} checks need attention.\nSee JSON for details.", COLORS["muted"], 8.7)

    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=DEFAULT_CHECKPOINT if DEFAULT_CHECKPOINT.exists() else RELEASE_CHECKPOINT)
    parser.add_argument("--guided", action="store_true", help="six presenter-led viva steps; Enter advances, q exits")
    parser.add_argument("--auto-advance", action="store_true", help="run guided steps without keyboard pauses (rehearsal)")
    parser.add_argument("--journey-seconds", type=float, default=90.0,
                        help="guided-mode live journey budget after annotated scenes (default: 90)")
    parser.add_argument("--model-id", default="B2_gat")
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--outage-duration", type=float, default=60.0)
    parser.add_argument("--demand-file", default=None)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default="cpu",
        help="inference device (default: cpu, the verified viva path; accelerators are opt-in)",
    )
    parser.add_argument("--gui", action="store_true", help="show SUMO while capturing")
    parser.add_argument(
        "--gui-step-delay-ms",
        type=int,
        default=100,
        help="wall-clock delay per SUMO step while GUI is active (default: 100)",
    )
    parser.add_argument(
        "--gui-simulation-step-s",
        type=float,
        default=0.25,
        help="SUMO time represented by each GUI frame (default: 0.25 seconds)",
    )
    parser.add_argument(
        "--gui-transition-seconds",
        type=float,
        default=1.0,
        help="ease-in/out camera transition time between scenes (default: 1)",
    )
    parser.add_argument(
        "--gui-phase-seconds",
        type=float,
        default=5.0,
        help="static display time for each annotated GUI phase (default: 5)",
    )
    parser.add_argument(
        "--gui-duration-seconds",
        type=float,
        default=600.0,
        help="total live GUI presentation time (default: 600 seconds)",
    )
    parser.add_argument(
        "--gui-window-size",
        nargs=2,
        type=int,
        default=(1440, 875),
        metavar=("WIDTH", "HEIGHT"),
        help="SUMO GUI window size (default: 1440 875 for this Mac display)",
    )
    parser.add_argument(
        "--close-gui-on-finish",
        action="store_true",
        help="close SUMO after the timed presentation instead of waiting",
    )
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "runs/viva_demo"
    )
    args = parser.parse_args()
    if args.auto_advance and not args.guided:
        parser.error("--auto-advance requires --guided")
    if args.guided and not args.auto_advance and not sys.stdin.isatty():
        parser.error("guided mode needs an interactive terminal; use --auto-advance for rehearsal")
    if not math.isfinite(args.journey_seconds) or args.journey_seconds < 0:
        parser.error("--journey-seconds must be finite and non-negative")
    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.max_steps < 1 or args.max_episodes < 1:
        parser.error("--max-steps and --max-episodes must be positive")
    if args.gui_step_delay_ms < 0:
        parser.error("--gui-step-delay-ms must be non-negative")
    if args.gui_simulation_step_s <= 0:
        parser.error("--gui-simulation-step-s must be positive")
    if args.gui_transition_seconds < 0:
        parser.error("--gui-transition-seconds must be non-negative")
    if args.gui_phase_seconds < 0:
        parser.error("--gui-phase-seconds must be non-negative")
    if args.gui_duration_seconds < 0:
        parser.error("--gui-duration-seconds must be non-negative")
    if any(size < 320 for size in args.gui_window_size):
        parser.error("--gui-window-size values must each be at least 320")
    return args


def main() -> int:
    args = _parse_args()
    _stage(args, 1, "System components and demonstration contract | 1 minute",
           "SUMO traffic -> DispatchEnv observations -> telemetry freeze -> GAT policy\n"
           "            -> sampled dispatch action -> SUMO execution\n"
           "Clean/degraded pair -> faithfulness evaluator -> offline release audit\n\n"
           "Implementation (under src/dispatch_marl/):\n"
           "  env.py                 traffic and dispatch environment\n"
           "  degradation.py         tunnel-triggered observation freeze\n"
           "  models/policy.py        graph policy and action outputs\n"
           "  faithfulness.py         paired perturbation-based evaluation\n"
           "  explanation_audit.py   formal explanation-release checks\n\n"
           "This run performs inference and paired scoring using a frozen checkpoint.\n"
           "The multi-checkpoint release verdict is read from the completed offline audit.\n"
           "Budget: roughly 10 minutes; Enter advances, q exits at a prompt, Ctrl+C stops.\n"
           + ("SUMO views also wait for Enter in this terminal." if args.gui else
              "Headless mode: model and metrics run live; no vehicle animation is shown."))
    _pause(args, "Load the model")
    _prepare_gui(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {args.checkpoint}")
    print(f"condition:  tunnel-triggered {args.outage_duration:g}s freeze")
    print("searching held-out demand for one stale-exposed decision ...")
    record, env = _capture_decision(args)
    try:
        json_path = args.output_dir / "viva_demo.json"
        png_path = args.output_dir / "viva_demo.png"
        json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        _render_figure(record, png_path)

        if args.guided:
            _stage(args, 3, "Model execution and dispatch | 3 minutes including GUI",
                   f"Captured agent: {record['scope']['agent']}; "
                   f"simulation time: {record['scope']['sim_time_s']:.2f} s\n"
                   "Actual batched input shapes:")
            execution = record["model_execution"]
            for name, shape in execution["input_shapes"].items():
                print(f"  {name}: {shape}")
            print(f"Output shapes: {execution['output_shapes']}")
            print("\nAction probabilities for the captured taxi (including padded slots):")
            for index, probability in enumerate(execution["action_probabilities"]):
                label = ("no-op" if index == 0 else
                         f"request {execution['reservation_ids'][index - 1]}"
                         if index <= len(execution["reservation_ids"]) else "masked/padded")
                print(f"  {index}: {label:<30} {probability:.6f}")
            print(f"Sampled action slot: {record['decision']['action_index']}; "
                  f"reservation ID: {record['decision']['reservation_id']}; "
                  f"P={record['decision']['probability']:.6f}")
            _pause(args, "Show SUMO execution" if args.gui else "Show paired outputs")
            if args.gui and record["scope"]["live_scene_matches_capture"]:
                _run_gui_presentation(record, args)
            elif args.gui:
                print("Historical fallback captured: live scene has advanced. "
                      "Skipping animation to avoid replaying an unrelated assignment.")
            _present_results(record, args)
            return 0

        print("\nCaptured decision")
        print(f"  agent/action: {record['scope']['agent']} / {record['decision']['action_label']}")
        print(f"  sim time:     {record['scope']['sim_time_s']:.0f} s")
        print(f"  stale node:   {record['telemetry']['focus_node_label']}")
        print(f"  AoI/error:    {record['telemetry']['max_aoi_s']:.0f} s / {record['telemetry']['position_error_m']:.1f} m")
        print(f"  WAMSN:        {record['explanation']['wamsn']:.3f}")
        print(f"  DEF:          {record['explanation']['primary_def']:+.3f} ({record['explanation']['primary_def_name']})")
        print(f"  audit:        {record['formal_audit']['model_decision']} (full offline evidence)")
        print(f"\nJSON:   {json_path}")
        print(f"Figure: {png_path}")
        if args.gui:
            print(f"Request: {args.output_dir / 'sumo_request_phase.png'}")
            print(f"Candidates: {args.output_dir / 'sumo_candidates_phase.png'}")
            print(f"Tunnel: {args.output_dir / 'sumo_tunnel_phase.png'}")
            print(f"Order:  {args.output_dir / 'sumo_order_phase.png'}")
            print(
                "Pickup/drop-off screenshots are added when those events occur."
            )
        if args.gui and record["scope"]["live_scene_matches_capture"]:
            _run_gui_presentation(record, args)
    except KeyboardInterrupt:
        print("\nDemo stopped by user.", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nDemo stopped by user.", flush=True)
