"""Tests for the streaming PerClassAnomalyAUROC node (one-vs-background BinaryAUROC per class)."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from torchmetrics.classification import BinaryAUROC

from cuvis_ai_dinomaly.node.per_class_auroc import PerClassAnomalyAUROC

CLASS_NAMES = {0: "bg", 1: "a", 2: "b"}


def _ctx(
    stage: ExecutionStage = ExecutionStage.TEST, epoch: int = 0, batch_idx: int = 0
) -> Context:
    return Context(stage=stage, epoch=epoch, batch_idx=batch_idx, global_step=batch_idx)


def _names(metrics) -> set[str]:
    return {m.name for m in metrics}


def test_composes_one_binary_auroc_per_non_background_class() -> None:
    """Reuses torchmetrics BinaryAUROC (one per class); background is excluded."""
    node = PerClassAnomalyAUROC(class_names=CLASS_NAMES, thresholds=128)
    assert set(node._aurocs.keys()) == {"1", "2"}
    assert all(isinstance(m, BinaryAUROC) for m in node._aurocs.values())


def test_forward_emits_per_class_running_metrics() -> None:
    """A batch with classes 1 and 2 present emits a running metric for each (not background)."""
    node = PerClassAnomalyAUROC(class_names=CLASS_NAMES)
    scores = torch.tensor([[[[0.1], [0.9]], [[0.8], [0.2]]]])  # [1, 2, 2, 1]
    class_mask = torch.tensor([[[[0], [1]], [[2], [0]]]], dtype=torch.int32)
    out = node.forward(scores=scores, class_mask=class_mask, context=_ctx())
    assert _names(out["metrics"]) == {"auroc_pixel_a", "auroc_pixel_b"}
    for m in out["metrics"]:
        assert m.stage == ExecutionStage.TEST and isinstance(m.value, float)


def test_perfect_separation_per_class() -> None:
    """Class-1 pixels scored high, background low -> one-vs-background AUROC ~1.0."""
    node = PerClassAnomalyAUROC(class_names={0: "bg", 1: "a"})
    scores = torch.tensor([[[[5.0], [5.0]], [[-5.0], [-5.0]]]])
    class_mask = torch.tensor([[[[1], [1]], [[0], [0]]]], dtype=torch.int32)
    out = node.forward(scores=scores, class_mask=class_mask, context=_ctx())
    assert {m.name: m.value for m in out["metrics"]}["auroc_pixel_a"] == pytest.approx(
        1.0, abs=1e-3
    )


def test_absent_class_omitted_from_compute() -> None:
    """A class never present in the masks is skipped by compute() (not 0.0/NaN)."""
    node = PerClassAnomalyAUROC(class_names=CLASS_NAMES)
    scores = torch.tensor([[[[0.9], [0.1]]]])  # [1, 1, 2, 1]
    class_mask = torch.tensor([[[[1], [0]]]], dtype=torch.int32)  # only class 1 + background
    node.forward(scores=scores, class_mask=class_mask, context=_ctx())
    computed = node.compute()
    assert "a" in computed and "b" not in computed


def test_compute_matches_pooled_sklearn_within_binning_tolerance() -> None:
    """Node's binned per-class AUROC tracks the exact pooled sklearn one-vs-background value."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    torch.manual_seed(0)
    node = PerClassAnomalyAUROC(class_names=CLASS_NAMES, thresholds=200)
    scores = torch.randn(1, 64, 64, 1)
    class_mask = torch.randint(0, 3, (1, 64, 64, 1), dtype=torch.int32)
    node.forward(scores=scores, class_mask=class_mask, context=_ctx())
    computed = node.compute()

    preds = torch.sigmoid(scores.flatten().float()).numpy()
    labels = class_mask.squeeze(-1).flatten().numpy()
    background = labels == 0
    for cid, name in ((1, "a"), (2, "b")):
        is_class = labels == cid
        keep = is_class | background
        expected = sklearn_metrics.roc_auc_score(is_class[keep].astype(int), preds[keep])
        assert computed[name] == pytest.approx(expected, abs=0.02)


def test_resets_on_stage_epoch_boundary() -> None:
    """A new (stage, epoch) restarts accumulation: an epoch seeing one label only yields 0.0,
    proving the separable prior-epoch state was cleared."""
    node = PerClassAnomalyAUROC(class_names={0: "bg", 1: "a"})
    # Epoch 0 — class 1 high, background low (perfect separation).
    scores = torch.tensor([[[[5.0], [-5.0]]]])
    class_mask = torch.tensor([[[[1], [0]]]], dtype=torch.int32)
    out0 = node.forward(scores=scores, class_mask=class_mask, context=_ctx(epoch=0))
    assert {m.name: m.value for m in out0["metrics"]}["auroc_pixel_a"] == pytest.approx(
        1.0, abs=1e-3
    )
    # Epoch 1 — class 1 only (no background). If state carried over it would stay ~1.0;
    # after reset it sees one label -> 0.0.
    scores = torch.tensor([[[[5.0], [5.0]]]])
    class_mask = torch.tensor([[[[1], [1]]]], dtype=torch.int32)
    out1 = node.forward(scores=scores, class_mask=class_mask, context=_ctx(epoch=1))
    assert {m.name: m.value for m in out1["metrics"]}["auroc_pixel_a"] == pytest.approx(
        0.0, abs=1e-6
    )
