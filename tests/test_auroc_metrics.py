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


# --- pooled epoch-end reduction (issue #6) -----------------------------------------------


def _spatial_batch(scores_hw: torch.Tensor, mask_hw: torch.Tensor):
    """Wrap [H, W] score + bool mask into the node's [B, H, W, 1] ports + an image score."""
    scores = scores_hw[None, :, :, None].float()
    targets = mask_hw[None, :, :, None].bool()
    anomaly_score = scores.flatten(1).max(dim=1).values
    return scores, targets, anomaly_score


def test_pooled_metric_names_declared() -> None:
    """The trainer keys off POOLED_METRIC_NAMES to skip per-batch logging of these names."""
    assert AnomalyAUROCMetrics.POOLED_METRIC_NAMES == frozenset({"auroc_pixel", "auroc_image"})


def test_pooled_metrics_empty_before_any_batch() -> None:
    """Nothing to log for a run that never produced scores."""
    assert AnomalyAUROCMetrics(thresholds=200).pooled_metrics() == {}


def test_pooled_compute_is_exact_not_per_batch_mean() -> None:
    """pooled_metrics() gives the exact pooled AUROC, not the biased per-batch mean.

    Batch A is an all-normal frame -> its running AUROC is undefined and torchmetrics
    returns 0.0, which poisons a per-batch mean (the issue-#6 failure). The pooled
    accumulator across A + B is unaffected.
    """
    torch.manual_seed(0)
    node = AnomalyAUROCMetrics(thresholds=200)

    a_scores, a_mask = torch.rand(8, 8), torch.zeros(8, 8, dtype=torch.bool)
    b_scores = torch.cat([torch.rand(4, 8) + 3.0, torch.rand(4, 8)], dim=0)
    b_mask = torch.zeros(8, 8, dtype=torch.bool)
    b_mask[:4, :] = True

    running = []
    for i, (s, m) in enumerate([(a_scores, a_mask), (b_scores, b_mask)]):
        out = node.forward(*_spatial_batch(s, m), context=_ctx(ExecutionStage.TEST, batch_idx=i))
        running.append({x.name: x.value for x in out["metrics"]}["auroc_pixel"])

    ref = BinaryAUROC(thresholds=200)
    ref.update(
        torch.sigmoid(torch.cat([a_scores.flatten(), b_scores.flatten()])),
        torch.cat([a_mask.flatten(), b_mask.flatten()]).long(),
    )

    pooled = node.pooled_metrics()
    assert set(pooled) == {"auroc_pixel", "auroc_image"}
    node_pooled = float(pooled["auroc_pixel"].compute())
    assert node_pooled == pytest.approx(float(ref.compute()), abs=1e-6)
    assert node_pooled > sum(running) / len(running) + 0.2


def test_pooled_metrics_reset_on_new_epoch() -> None:
    """The (stage, epoch) boundary clears the pooled accumulator."""
    node = AnomalyAUROCMetrics(thresholds=200)
    s = torch.cat([torch.rand(4, 8) + 3.0, torch.rand(4, 8)], dim=0)
    m = torch.zeros(8, 8, dtype=torch.bool)
    m[:4, :] = True
    node.forward(*_spatial_batch(s, m), context=_ctx(ExecutionStage.TEST, epoch=0))
    first = float(node.pooled_metrics()["auroc_pixel"].compute())
    node.forward(*_spatial_batch(s, m), context=_ctx(ExecutionStage.TEST, epoch=1))
    assert float(node.pooled_metrics()["auroc_pixel"].compute()) == pytest.approx(first, abs=1e-6)
