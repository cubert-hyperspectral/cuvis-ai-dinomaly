"""Tests for the streaming PerClassAnomalyAUROC node (one-vs-background BinaryAUROC per class)."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
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


def test_category_mask_matches_class_mask() -> None:
    """The cu3s shape of the class ids ([B, H, W] in the batch mask) gives the same AUROC."""
    torch.manual_seed(0)
    scores = torch.randn(2, 16, 16, 1)
    category_mask = torch.randint(0, 3, (2, 16, 16), dtype=torch.int32)
    from_class_mask = PerClassAnomalyAUROC(class_names=CLASS_NAMES)
    from_class_mask.forward(scores=scores, class_mask=category_mask.unsqueeze(-1), context=_ctx())
    from_category_mask = PerClassAnomalyAUROC(class_names=CLASS_NAMES)
    from_category_mask.forward(scores=scores, category_mask=category_mask, context=_ctx())
    assert from_category_mask.compute() == from_class_mask.compute()


def test_without_class_ids_raises() -> None:
    """Neither class_mask nor category_mask connected is a wiring error, not an empty result."""
    node = PerClassAnomalyAUROC(class_names=CLASS_NAMES)
    with pytest.raises(ValueError, match="category_mask"):
        node.forward(scores=torch.zeros(1, 2, 2, 1), context=_ctx())


def test_both_class_id_ports_raise() -> None:
    """Both connected (a graph half moved from npz_multi) is a wiring error, not a silent pick."""
    node = PerClassAnomalyAUROC(class_names=CLASS_NAMES)
    ids = torch.zeros(1, 2, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match="both"):
        node.forward(
            scores=torch.zeros(1, 2, 2, 1),
            context=_ctx(),
            class_mask=ids.unsqueeze(-1),
            category_mask=ids,
        )


def test_category_mask_port_accepts_cu3s_data_node_mask() -> None:
    """The mask output of cuvis-ai's CU3SDataNode ([B, H, W] int32) binds to category_mask;
    class_mask keeps its [B, H, W, 1] contract, and both ports are optional."""
    cu3s_data_node_mask = PortSpec(dtype=torch.int32, shape=(-1, -1, -1), description="mask")
    specs = PerClassAnomalyAUROC.INPUT_SPECS
    ok, message = cu3s_data_node_mask.is_compatible_with(specs["category_mask"], None, None)
    assert ok, message
    ok, _ = cu3s_data_node_mask.is_compatible_with(specs["class_mask"], None, None)
    assert not ok
    assert specs["class_mask"].optional and specs["category_mask"].optional
