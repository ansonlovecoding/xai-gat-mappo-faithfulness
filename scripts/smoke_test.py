"""Smoke test for the SUMO + TraCI + sumolib pipeline.

Generates a tiny 3x3 grid scenario, runs it through TraCI, and asserts that
vehicles actually flow. Catches broken SUMO_HOME wiring, version mismatches,
and PATH issues before any real env code is written.

Run from project root with the venv active:
    python scripts/smoke_test.py              # headless (fast)
    python scripts/smoke_test.py --gui        # open sumo-gui, auto-play
    python scripts/smoke_test.py --gui --delay 200   # slower playback (ms/step)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = PROJECT_ROOT / "scenarios" / "smoke"

load_dotenv(PROJECT_ROOT / ".env")

# Candidate SUMO_HOME locations, in priority order. Resolved BEFORE importing
# sumolib/traci so the bindings find the right tools/data dir even when the
# script is launched directly (IDE run, cron, etc.) and `activate` was never
# sourced.
_SUMO_HOME_CANDIDATES = [
    "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo",
    "/opt/homebrew/share/sumo",
    "/usr/local/share/sumo",
    "/usr/share/sumo",
]


def _resolve_sumo_home() -> str:
    env_value = os.environ.get("SUMO_HOME")
    if env_value and Path(env_value, "tools").is_dir():
        return env_value
    for candidate in _SUMO_HOME_CANDIDATES:
        if Path(candidate, "tools").is_dir():
            os.environ["SUMO_HOME"] = candidate
            bin_dir = str(Path(candidate, "bin"))
            if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return candidate
    raise RuntimeError(
        "Could not locate SUMO. Set SUMO_HOME or install SUMO. "
        f"Checked: {_SUMO_HOME_CANDIDATES}"
    )


_resolve_sumo_home()

import sumolib  # noqa: E402  — must follow SUMO_HOME resolution
import traci  # noqa: E402


def generate_scenario() -> Path:
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    net = SCENARIO_DIR / "grid.net.xml"
    rou = SCENARIO_DIR / "grid.rou.xml"
    cfg = SCENARIO_DIR / "grid.sumocfg"

    netgenerate = shutil.which("netgenerate") or str(
        Path(os.environ["SUMO_HOME"], "bin", "netgenerate")
    )
    subprocess.run(
        [
            netgenerate,
            "--grid",
            "--grid.number=3",
            "--grid.length=100",
            "--output-file",
            str(net),
        ],
        check=True,
    )

    random_trips = Path(os.environ["SUMO_HOME"]) / "tools" / "randomTrips.py"
    subprocess.run(
        [
            sys.executable,
            str(random_trips),
            "-n", str(net),
            "-r", str(rou),
            "-e", "100",
            "--seed", "42",
        ],
        check=True,
    )

    cfg.write_text(
        f"""<configuration>
  <input>
    <net-file value="{net.name}"/>
    <route-files value="{rou.name}"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="200"/>
  </time>
</configuration>
"""
    )
    return cfg


def run_traci(cfg: Path, steps: int = 200, gui: bool = False, delay_ms: int = 100) -> dict:
    sumo_binary = sumolib.checkBinary("sumo-gui" if gui else "sumo")
    cmd = [sumo_binary, "-c", str(cfg), "--no-step-log", "--no-warnings"]
    if gui:
        cmd += ["--start", "--quit-on-end", "--delay", str(delay_ms)]
    traci.start(cmd)
    try:
        total_departed = 0
        max_concurrent = 0
        for _ in range(steps):
            traci.simulationStep()
            total_departed += traci.simulation.getDepartedNumber()
            max_concurrent = max(max_concurrent, traci.vehicle.getIDCount())
        return {
            "total_departed": total_departed,
            "max_concurrent": max_concurrent,
        }
    finally:
        traci.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="open sumo-gui instead of headless sumo")
    parser.add_argument("--delay", type=int, default=100, help="ms delay per step in GUI mode (default 100)")
    args = parser.parse_args()

    print(f"SUMO_HOME = {os.environ['SUMO_HOME']}")
    cfg = generate_scenario()
    print(f"Generated scenario: {cfg}")

    stats = run_traci(cfg, gui=args.gui, delay_ms=args.delay)
    print(f"Simulation stats: {stats}")

    assert stats["total_departed"] > 0, (
        "no vehicles departed — scenario or SUMO is broken"
    )
    print("OK — SUMO end-to-end pipeline is healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
