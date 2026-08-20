"""Tests for the label-free ValNormalAnomalyMean monitoring node."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

from cuvis_ai_dinomaly.node.val_anomaly_mean import ValNormalAnomalyMean


def _ctx(stage: ExecutionStage = ExecutionStage.VAL, epoch: int = 0, batch_idx: int = 0) -> Context:
    return Context(stage=stage, epoch=epoch, batch_idx=batch_idx, global_step=batch_idx)


def _mean(out: dict) -> float:
    (metric,) = out["metrics"]
    assert metric.name == "mean_anomaly_normal"
    return metric.value


def test_running_mean_accumulates_across_batches() -> None:
    """Each forward emits the running mean over all batches of the (stage, epoch)."""
    node = ValNormalAnomalyMean()
    out = node.forward(anomaly_score=torch.tensor([0.2, 0.4]), context=_ctx(batch_idx=0))
    assert _mean(out) == pytest.approx(0.3)
    out = node.forward(anomaly_score=torch.tensor([0.6]), context=_ctx(batch_idx=1))
    assert _mean(out) == pytest.approx(0.4)  # (0.2 + 0.4 + 0.6) / 3


def test_resets_on_stage_epoch_boundary() -> None:
    """A new (stage, epoch) restarts accumulation: the first batch of epoch 1 yields
    its own mean, proving prior-epoch state was cleared."""
    node = ValNormalAnomalyMean()
    node.forward(anomaly_score=torch.tensor([0.2, 0.4]), context=_ctx(epoch=0, batch_idx=0))
    node.forward(anomaly_score=torch.tensor([0.6]), context=_ctx(epoch=0, batch_idx=1))
    out = node.forward(anomaly_score=torch.tensor([0.8, 1.0]), context=_ctx(epoch=1, batch_idx=0))
    assert _mean(out) == pytest.approx(0.9)  # not (0.2 + 0.4 + 0.6 + 0.8 + 1.0) / 5


def test_explicit_reset_clears_state() -> None:
    node = ValNormalAnomalyMean()
    node.forward(anomaly_score=torch.tensor([0.2, 0.4]), context=_ctx(batch_idx=0))
    node.reset()
    out = node.forward(anomaly_score=torch.tensor([0.6]), context=_ctx(batch_idx=0))
    assert _mean(out) == pytest.approx(0.6)  # not (0.2 + 0.4 + 0.6) / 3


def test_stage_filter_val_test_only() -> None:
    node = ValNormalAnomalyMean()
    assert set(node.execution_stages) == {ExecutionStage.VAL, ExecutionStage.TEST}
