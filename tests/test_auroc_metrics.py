"""Tests for the streaming AnomalyAUROCMetrics node (torchmetrics BinaryAUROC)."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from torchmetrics.classification import BinaryAUROC

from cuvis_ai_dinomaly.node.auroc_metrics import AnomalyAUROCMetrics


def _ctx(stage: ExecutionStage = ExecutionStage.VAL, epoch: int = 0, batch_idx: int = 0) -> Context:
    return Context(stage=stage, epoch=epoch, batch_idx=batch_idx, global_step=batch_idx)


def _batch(b: int, h: int, w: int, score_value: float, all_anomaly: bool):
    scores = torch.full((b, h, w, 1), score_value)
    targets = torch.full((b, h, w, 1), all_anomaly, dtype=torch.bool)
    anomaly_score = torch.full((b,), score_value)
    return scores, targets, anomaly_score


def _names(metrics) -> set[str]:
    return {m.name for m in metrics}


def test_forward_emits_running_auroc_metrics() -> None:
    """Each forward emits running auroc_pixel + auroc_image as Metric objects (no callback)."""
    node = AnomalyAUROCMetrics()
    s, t, a = _batch(1, 4, 4, score_value=0.5, all_anomaly=True)
    out = node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx())
    assert _names(out["metrics"]) == {"auroc_pixel", "auroc_image"}
    for m in out["metrics"]:
        assert m.stage == ExecutionStage.VAL and isinstance(m.value, float)


def test_running_auroc_perfect_separation() -> None:
    """After a both-classes epoch with separable scores, the running AUROC is ~1.0."""
    node = AnomalyAUROCMetrics()
    s, t, a = _batch(1, 4, 4, score_value=5.0, all_anomaly=True)  # positives, high score
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(batch_idx=0))
    s, t, a = _batch(1, 4, 4, score_value=-5.0, all_anomaly=False)  # negatives, low score
    out = node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(batch_idx=1))
    vals = {m.name: m.value for m in out["metrics"]}
    assert vals["auroc_pixel"] == pytest.approx(1.0, abs=1e-3)
    assert vals["auroc_image"] == pytest.approx(1.0, abs=1e-3)


def test_resets_on_stage_epoch_boundary() -> None:
    """A new (stage, epoch) restarts accumulation: a fresh epoch seeing only one class
    yields AUROC 0.0 (undefined), proving prior-epoch state was cleared."""
    node = AnomalyAUROCMetrics()
    # Epoch 0 — both classes (perfect separation).
    s, t, a = _batch(1, 4, 4, score_value=5.0, all_anomaly=True)
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(epoch=0, batch_idx=0))
    s, t, a = _batch(1, 4, 4, score_value=-5.0, all_anomaly=False)
    out0 = node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(epoch=0, batch_idx=1))
    assert {m.name: m.value for m in out0["metrics"]}["auroc_pixel"] == pytest.approx(1.0, abs=1e-3)
    # Epoch 1 — single all-positive batch. If state carried over, AUROC would be ~1.0;
    # after reset it sees one class only -> 0.0.
    s, t, a = _batch(1, 4, 4, score_value=5.0, all_anomaly=True)
    out1 = node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(epoch=1, batch_idx=0))
    assert {m.name: m.value for m in out1["metrics"]}["auroc_pixel"] == pytest.approx(0.0, abs=1e-6)


def test_explicit_reset_clears_state() -> None:
    node = AnomalyAUROCMetrics()
    s, t, a = _batch(1, 4, 4, score_value=5.0, all_anomaly=True)
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(batch_idx=0))
    s, t, a = _batch(1, 4, 4, score_value=-5.0, all_anomaly=False)
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(batch_idx=1))
    node.reset()
    # After reset, a single one-class batch -> AUROC 0.0 (undefined), not the prior 1.0.
    s, t, a = _batch(1, 4, 4, score_value=5.0, all_anomaly=True)
    out = node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(batch_idx=0))
    assert {m.name: m.value for m in out["metrics"]}["auroc_pixel"] == pytest.approx(0.0, abs=1e-6)


def test_streaming_state_is_bounded_no_cpu_concat() -> None:
    """Regression for the memory fix: state is torchmetrics BinaryAUROC (O(thresholds)),
    not the old unbounded per-pixel CPU lists."""
    node = AnomalyAUROCMetrics(thresholds=128)
    assert isinstance(node.pixel_auroc, BinaryAUROC)
    assert isinstance(node.image_auroc, BinaryAUROC)
    assert not hasattr(node, "_pixel_preds")  # the old couple-GB-per-epoch concat is gone
    # Feeding many large batches must not grow any Python-side buffer.
    for i in range(8):
        s, t, a = _batch(1, 64, 64, score_value=float(i), all_anomaly=bool(i % 2))
        node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(batch_idx=i))
    # No list attribute should be accumulating tensors.
    assert not any(isinstance(v, list) and v for v in vars(node).values())


def test_stage_filter_val_test_only() -> None:
    node = AnomalyAUROCMetrics()
    assert ExecutionStage.VAL in node.execution_stages
    assert ExecutionStage.TEST in node.execution_stages
    assert ExecutionStage.TRAIN not in node.execution_stages
