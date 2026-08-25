"""Deterministic random-number setup for training and evaluation."""
from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
import torch


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable 31-bit seed from a base seed and semantic labels."""
    payload = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> dict[str, Any]:
    """Seed Python, NumPy, and PyTorch and return the applied settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    return {
        "seed": int(seed),
        "python": int(seed),
        "numpy_legacy": int(seed),
        "torch": int(seed),
        "deterministic_torch": bool(deterministic_torch),
    }
