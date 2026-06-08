"""Tests for AnomalyAUROCMetrics + AUROCEpochEndCallback (dinomaly2-style)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

from cuvis_ai_dinomaly.node.auroc_metrics import AnomalyAUROCMetrics, AUROCEpochEndCallback


def _ctx(stage: ExecutionStage = ExecutionStage.VAL, epoch: int = 0, batch_idx: int = 0) -> Context:
    return Context(stage=stage, epoch=epoch, batch_idx=batch_idx, global_step=batch_idx)


def _batch(b: int, h: int, w: int, score_value: float, all_anomaly: bool):
    scores = torch.full((b, h, w, 1), score_value)
    targets = torch.full((b, h, w, 1), all_anomaly, dtype=torch.bool)
    anomaly_score = torch.full((b,), score_value)
    return scores, targets, anomaly_score


def test_forward_returns_empty_metrics_list() -> None:
    node = AnomalyAUROCMetrics()
    scores, targets, anom = _batch(1, 4, 4, score_value=0.5, all_anomaly=True)
    out = node.forward(scores=scores, targets=targets, anomaly_score=anom, context=_ctx())
    assert out == {"metrics": []}


def test_compute_auroc_perfect_separation() -> None:
    node = AnomalyAUROCMetrics()
    # Frame 0 — all positive GT, high score.
    s, t, a = _batch(1, 4, 4, score_value=5.0, all_anomaly=True)
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx())
    # Frame 1 — all background, low score.
    s, t, a = _batch(1, 4, 4, score_value=-5.0, all_anomaly=False)
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(batch_idx=1))
    out = node.compute_auroc()
    assert out["auroc_pixel"] == pytest.approx(1.0, abs=1e-6)
    assert out["auroc_image"] == pytest.approx(1.0, abs=1e-6)


def test_compute_auroc_skipped_when_all_targets_same_class() -> None:
    node = AnomalyAUROCMetrics()
    s, t, a = _batch(2, 4, 4, score_value=1.0, all_anomaly=True)
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx())
    out = node.compute_auroc()
    # Both pixels and images are all-positive → AUROC undefined → key absent.
    assert "auroc_pixel" not in out
    assert "auroc_image" not in out


def test_compute_auroc_empty_when_no_data() -> None:
    node = AnomalyAUROCMetrics()
    assert node.compute_auroc() == {}


def test_reset_clears_accumulators() -> None:
    node = AnomalyAUROCMetrics()
    s, t, a = _batch(1, 4, 4, score_value=1.0, all_anomaly=True)
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx())
    assert node._pixel_preds  # populated
    node.reset()
    assert not node._pixel_preds
    assert not node._pixel_targets
    assert not node._image_preds
    assert not node._image_targets


def test_stage_filter_val_test_only() -> None:
    node = AnomalyAUROCMetrics()
    assert ExecutionStage.VAL in node.execution_stages
    assert ExecutionStage.TEST in node.execution_stages
    assert ExecutionStage.TRAIN not in node.execution_stages


def test_callback_logs_auroc_on_val_epoch_end_with_pl_module_log() -> None:
    """The callback must (a) reset the node at epoch start and (b) call
    pl_module.log(<prefix>/auroc_*) on epoch end with the values computed from
    whatever the node accumulated during forward passes."""
    node = AnomalyAUROCMetrics()
    cb = AUROCEpochEndCallback(auroc_node=node, node_log_prefix="metrics_auroc")

    trainer = MagicMock()
    pl_module = MagicMock()

    # epoch start → reset (no logging)
    cb.on_validation_epoch_start(trainer, pl_module)
    assert pl_module.log.call_count == 0

    # Accumulate a perfect-separation epoch.
    s, t, a = _batch(1, 4, 4, score_value=5.0, all_anomaly=True)
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx())
    s, t, a = _batch(1, 4, 4, score_value=-5.0, all_anomaly=False)
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(batch_idx=1))

    cb.on_validation_epoch_end(trainer, pl_module)
    # Two metrics → two log() calls with prog_bar=True / on_epoch=True.
    logged_names = {call.args[0] for call in pl_module.log.call_args_list}
    assert logged_names == {"metrics_auroc/auroc_pixel", "metrics_auroc/auroc_image"}
    for call in pl_module.log.call_args_list:
        assert call.kwargs.get("prog_bar") is True
        assert call.kwargs.get("on_epoch") is True
        assert call.args[1] == pytest.approx(1.0, abs=1e-6)


def test_callback_resets_on_epoch_start_between_epochs() -> None:
    """Accumulator must be cleared on each new validation epoch."""
    node = AnomalyAUROCMetrics()
    cb = AUROCEpochEndCallback(auroc_node=node)
    trainer = MagicMock()
    pl_module = MagicMock()

    # Epoch 0 — populate.
    s, t, a = _batch(2, 4, 4, score_value=1.0, all_anomaly=True)
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(epoch=0))
    assert node._pixel_preds

    # Epoch 1 — callback fires epoch_start, accumulator cleared.
    cb.on_validation_epoch_start(trainer, pl_module)
    assert not node._pixel_preds
