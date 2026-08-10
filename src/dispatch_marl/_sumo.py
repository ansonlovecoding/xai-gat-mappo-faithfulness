"""Locate SUMO_HOME once, and pick the fastest available Python backend.

Two responsibilities:

1. Resolve `SUMO_HOME` on import (SUMO looks up its data via this env var).
2. Import the SUMO Python bindings, preferring `libsumo` (which runs SUMO
   in-process, ~5-10× faster than the socket-based `traci`), and falling
   back to `traci` if libsumo isn't available.

Every module inside this package that needs a `traci`-shaped API should
`from ._sumo import traci` — do not `import traci` directly, otherwise
you bypass the libsumo speedup.

`USING_LIBSUMO` is exposed so callers can branch on backend when the two
differ semantically (only real difference at time of writing: libsumo has
no GUI support).
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


def _pypi_wheel_candidate() -> str | None:
    """The `eclipse-sumo` PyPI wheel ships the full SUMO distribution as an
    importable `sumo` package (bin/ + tools/ + data/ inside site-packages).
    Resolving it here makes the repo work on any machine where the wheel is
    installed, without a system-level SUMO."""
    try:
        import sumo  # type: ignore[import-not-found]
        return str(Path(sumo.__file__).parent)
    except ImportError:
        return None


def resolve_sumo_home() -> Path:
    env_value = os.environ.get("SUMO_HOME")
    if env_value and Path(env_value, "tools").is_dir():
        return Path(env_value)
    candidates = list(_SUMO_HOME_CANDIDATES)
    wheel = _pypi_wheel_candidate()
    if wheel is not None:
        candidates.append(wheel)
    for candidate in candidates:
        if Path(candidate, "tools").is_dir():
            os.environ["SUMO_HOME"] = candidate
            bin_dir = str(Path(candidate, "bin"))
            if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return Path(candidate)
    raise RuntimeError(
        f"SUMO not found. Set SUMO_HOME or install SUMO. Checked: {candidates}"
    )


SUMO_HOME = resolve_sumo_home()


# ---- Backend selection -----------------------------------------------------

def _load_backend():
    """Try libsumo first, fall back to traci."""
    if os.environ.get("DISPATCH_MARL_FORCE_TRACI") == "1":
        import traci as backend
        return backend, False
    try:
        import libsumo as backend
        # libsumo has no `switch` (single in-process connection, so no need
        # to route commands to a specific connection). Env code uses switch
        # to disambiguate multiple resets, so give it a no-op stand-in and
        # let the same env code path work on both backends.
        if not hasattr(backend, "switch"):
            backend.switch = lambda label: None
        return backend, True
    except ImportError:
        import traci as backend
        return backend, False


traci, USING_LIBSUMO = _load_backend()
