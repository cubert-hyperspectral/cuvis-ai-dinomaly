"""Contract checks for published node specs and constructor config surface."""

from __future__ import annotations

import torch

from cuvis_ai_dinomaly.node.dinomaly_detector import DinomalyDetector
from cuvis_ai_dinomaly.node.dinomaly_train_loss_bridge import DinomalyTrainLossBridge


def test_detector_port_specs_contract() -> None:
    ins = DinomalyDetector.INPUT_SPECS
    outs = DinomalyDetector.OUTPUT_SPECS

    assert set(ins.keys()) == {"rgb_image"}
    assert ins["rgb_image"].dtype == torch.float32
    assert ins["rgb_image"].shape == (-1, -1, -1, 3)

    assert {"scores", "anomaly_score", "training_loss"} <= set(outs.keys())
    assert outs["scores"].shape == (-1, -1, -1, 1)
    assert outs["anomaly_score"].shape == (-1,)
    assert outs["training_loss"].shape == ()
    assert outs["training_loss"].optional is True


def test_loss_bridge_port_specs_contract() -> None:
    ins = DinomalyTrainLossBridge.INPUT_SPECS
    outs = DinomalyTrainLossBridge.OUTPUT_SPECS
    assert set(ins.keys()) == {"raw_loss"}
    assert ins["raw_loss"].shape == ()
    assert ins["raw_loss"].optional is True
    assert set(outs.keys()) == {"loss"}
    assert outs["loss"].shape == ()
