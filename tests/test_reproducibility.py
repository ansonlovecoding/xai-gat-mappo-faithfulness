import random

import numpy as np
import torch

from dispatch_marl.reproducibility import derive_seed, seed_everything


def _draws(seed: int) -> tuple[float, float, float]:
    seed_everything(seed)
    return random.random(), float(np.random.random()), float(torch.rand(()))


def test_seed_everything_repeats_all_random_streams() -> None:
    assert _draws(42) == _draws(42)
    assert _draws(42) != _draws(43)


def test_derived_seeds_are_stable_and_semantic() -> None:
    assert derive_seed(42, "evaluation", 0) == derive_seed(42, "evaluation", 0)
    assert derive_seed(42, "evaluation", 0) != derive_seed(42, "evaluation", 1)
