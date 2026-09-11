import pytest
from dispatch_marl.runtime import choose_device


def test_explicit_cpu_overrides_available_accelerators(monkeypatch):
    monkeypatch.setenv("DISPATCH_MARL_DEVICE", "cpu")
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert choose_device() == "cpu"


def test_invalid_device_is_rejected(monkeypatch):
    monkeypatch.setenv("DISPATCH_MARL_DEVICE", "invalid")
    with pytest.raises(ValueError, match="DISPATCH_MARL_DEVICE"):
        choose_device()
