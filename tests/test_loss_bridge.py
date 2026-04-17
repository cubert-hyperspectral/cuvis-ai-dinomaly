import pytest
import torch

from cuvis_ai_dinomaly.node.dinomaly_train_loss_bridge import DinomalyTrainLossBridge


def test_bridge_scales_loss() -> None:
    bridge = DinomalyTrainLossBridge(weight=2.0, name="t")
    x = torch.tensor(0.5, dtype=torch.float32)
    out = bridge.forward(raw_loss=x)
    assert out["loss"].item() == pytest.approx(1.0)


def test_bridge_flattens_non_scalar_raw_loss() -> None:
    bridge = DinomalyTrainLossBridge(weight=1.0, name="t")
    x = torch.tensor([[0.25]], dtype=torch.float32)
    out = bridge.forward(raw_loss=x)
    assert out["loss"].shape == ()
    assert out["loss"].item() == pytest.approx(0.25)


def test_bridge_handles_missing_raw_loss_in_eval() -> None:
    bridge = DinomalyTrainLossBridge(weight=1.0, name="t")
    out = bridge.forward(raw_loss=None)
    assert out["loss"].shape == ()
    assert out["loss"].item() == pytest.approx(0.0)


def test_bridge_preserves_input_dtype() -> None:
    bridge = DinomalyTrainLossBridge(weight=1.0, name="t")
    x = torch.tensor(0.25, dtype=torch.float64)
    out = bridge.forward(raw_loss=x)
    assert out["loss"].dtype == torch.float64


def test_bridge_preserves_device_for_present_loss() -> None:
    bridge = DinomalyTrainLossBridge(weight=1.0, name="t")
    x = torch.tensor(0.5, dtype=torch.float32)
    out = bridge.forward(raw_loss=x)
    assert out["loss"].device == x.device
