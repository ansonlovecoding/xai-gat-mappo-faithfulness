"""Locate SUMO_HOME once, on import.

Centralises the copy-pasted resolution block from `scripts/*.py`. Every module
in this package that needs `sumolib` or `traci` should import from here.
"""
from __future__ import annotations

import os
from pathlib import Path

_SUMO_HOME_CANDIDATES = [
    "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo",
    "/opt/homebrew/share/sumo",
    "/usr/local/share/sumo",
    "/usr/share/sumo",
]


def resolve_sumo_home() -> Path:
    env_value = os.environ.get("SUMO_HOME")
    if env_value and Path(env_value, "tools").is_dir():
        return Path(env_value)
    for candidate in _SUMO_HOME_CANDIDATES:
        if Path(candidate, "tools").is_dir():
            os.environ["SUMO_HOME"] = candidate
            bin_dir = str(Path(candidate, "bin"))
            if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return Path(candidate)
    raise RuntimeError(
        f"SUMO not found. Set SUMO_HOME or install SUMO. Checked: {_SUMO_HOME_CANDIDATES}"
    )


SUMO_HOME = resolve_sumo_home()
