"""Shared device selection and checkpoint reconstruction."""
from __future__ import annotations

import platform
from pathlib import Path

import torch

from .models import (
    DispatchGATPolicy,
    DispatchMLPPolicy,
    MLPPolicyConfig,
    PolicyConfig,
)


def choose_device() -> str:
    if torch.backends.mps.is_available() and platform.machine() == "arm64":
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_policy(checkpoint_path: Path, device: str):
    """Reconstruct a GAT or MLP policy from a training checkpoint."""
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    policy_type = checkpoint.get("policy_type", "gat")
    if policy_type == "mlp":
        config = MLPPolicyConfig(
            **{**checkpoint["policy_config"], "device": device}
        )
        policy = DispatchMLPPolicy(config)
    elif policy_type == "gat":
        config = PolicyConfig(
            **{**checkpoint["policy_config"], "device": device}
        )
        policy = DispatchGATPolicy(config)
    else:
        raise ValueError(f"unsupported checkpoint policy_type: {policy_type!r}")
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    return policy, checkpoint
