"""Streaming pixel-AUROC + image-AUROC for Dinomaly (validation / test).

Mirrors the :class:`cuvis_ai.node.metrics.AnomalyDetectionMetrics` pattern: a
``torchmetrics`` ``BinaryAUROC`` with histogram ``thresholds`` (so per-epoch state is
O(thresholds), not the couple-GB-per-epoch CPU concat of every pixel) is accumulated
via ``update()`` across batches and reset on the ``(stage, epoch)`` boundary. Each
forward emits the *running* AUROC as a :class:`~cuvis_ai_schemas.execution.Metric`. No
bespoke Lightning callback is needed.

The per-batch ``Metric.value`` emitted by ``forward`` is a *running* AUROC — a
batch-size-sensitive approximation if mean-reduced over the epoch. The authoritative epoch
value comes from :meth:`pooled_metrics`: the node lists ``auroc_pixel`` / ``auroc_image`` in
``POOLED_METRIC_NAMES``, so the trainer skips their per-batch float logging and instead logs
the live ``BinaryAUROC`` objects with ``on_epoch=True``, and Lightning does one pooled
``compute()`` + ``reset()`` at epoch end — exact and batch-size-invariant. This mirrors
``cuvis_ai.node.metrics.AnomalyDetectionMetrics`` and closes the reporting gap tracked in
issue #6. The per-batch values remain available on the ``metrics`` port for live monitoring
(e.g. the TensorBoard node).

Scores are passed through ``sigmoid`` before the binned metric so the thresholds span
``[0, 1]``; AUROC is rank-invariant under a monotonic transform, so the value is
unchanged.

UPSTREAM USAGE (deferred move): the long-term home is ``cuvis-ai``
``cuvis_ai/node/metrics.py``, next to ``AnomalyDetectionMetrics`` (whose streaming
pattern this mirrors), so *any* pipeline — not just Dinomaly — can wire a streaming
pixel/image AUROC node without depending on this plugin. Two gates before moving it:

1. It needs a cuvis-ai release shipping the node at the upstream import path, then a
   re-point of already-saved pipelines (incl. the published HF bedding model) whose YAML
   references ``cuvis_ai_dinomaly.node.auroc_metrics.AnomalyAUROCMetrics`` — the same
   re-point dance as the selector retirement (cuvis-ai#39).
2. Ideally land it with a proper epoch-level reduction upstream (an epoch-end hook, or a
   last-batch flag on ``Context``) so the reported epoch value is the exact pooled AUROC
   rather than the per-batch mean noted above.

Until then it stays here as a plugin-local monitoring metric.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context, Metric
from cuvis_ai_schemas.pipeline import PortSpec
from torchmetrics.classification import BinaryAUROC

from cuvis_ai_dinomaly.node._binned_auroc import _StreamingBinnedAUROC

Tensor = torch.Tensor


class AnomalyAUROCMetrics(_StreamingBinnedAUROC):
    """Streaming pixel/image AUROC via torchmetrics (val/test only).

    Input ports
    -----------
    scores : ``[B, H, W, 1] float32`` — raw anomaly map (not thresholded)
    targets : ``[B, H, W, 1] bool``   — ground-truth pixel masks
    anomaly_score : ``[B] float32``    — per-image score (top-k mean of amap)

    Output ports
    ------------
    metrics : ``list[Metric]`` — running ``auroc_pixel`` / ``auroc_image`` per batch (for
        live monitoring, e.g. the TensorBoard node). The authoritative epoch value is the
        pooled ``compute()`` the trainer logs from :meth:`pooled_metrics` at epoch end.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.ANOMALY})

    # auroc_pixel / auroc_image accumulate across the whole epoch and must be reduced by a
    # single pooled compute() at epoch end, not by averaging the per-batch running values
    # (batch-size-sensitive; badly biased at batch_size=1). The trainer skips these names in
    # per-batch logging and instead logs the live torchmetrics objects from pooled_metrics()
    # with on_epoch=True, so Lightning does the pooled compute()+reset() natively. Mirrors
    # cuvis_ai.node.metrics.AnomalyDetectionMetrics.
    POOLED_METRIC_NAMES: ClassVar[frozenset[str]] = frozenset({"auroc_pixel", "auroc_image"})

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Raw anomaly map [B, H, W, 1]",
        ),
        "targets": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Ground-truth pixel masks [B, H, W, 1]",
        ),
        "anomaly_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Per-image anomaly score [B]",
        ),
    }
    OUTPUT_SPECS = {
        "metrics": PortSpec(
            dtype=list, shape=(), description="List of Metric objects (running AUROC)"
        ),
    }

    def __init__(self, thresholds: int = 200, **kwargs: Any) -> None:
        super().__init__(thresholds=thresholds, **kwargs)
        # Histogram-based AUROC: O(thresholds) state, accumulated across batches and reset
        # only at the (stage, epoch) boundary. forward() emits the running value per batch;
        # the pooled epoch value is logged via pooled_metrics() (see POOLED_METRIC_NAMES).
        self.pixel_auroc = BinaryAUROC(thresholds=thresholds)
        self.image_auroc = BinaryAUROC(thresholds=thresholds)

    def _reset_state(self) -> None:
        self.pixel_auroc.reset()
        self.image_auroc.reset()

    def forward(
        self,
        scores: Tensor,
        targets: Tensor,
        anomaly_score: Tensor,
        context: Context,
    ) -> dict[str, Any]:
        # Reset on the (stage, epoch) boundary so each epoch accumulates fresh.
        self._reset_on_epoch_boundary(context)

        # Pixel-level: sigmoid -> [0, 1] for the binned metric (AUROC is rank-invariant).
        self.pixel_auroc.update(self._binned_preds(scores), targets.squeeze(-1).flatten().long())

        # Image-level: per-image score vs "any GT pixel positive" label.
        img_tgts = targets.squeeze(-1).flatten(1).any(dim=1).long()
        self.image_auroc.update(self._binned_preds(anomaly_score), img_tgts)

        return {
            "metrics": [
                Metric(
                    name="auroc_pixel",
                    value=float(self.pixel_auroc.compute()),
                    stage=context.stage,
                    epoch=context.epoch,
                    batch_idx=context.batch_idx,
                ),
                Metric(
                    name="auroc_image",
                    value=float(self.image_auroc.compute()),
                    stage=context.stage,
                    epoch=context.epoch,
                    batch_idx=context.batch_idx,
                ),
            ]
        }

    def pooled_metrics(self) -> dict[str, BinaryAUROC]:
        """Live torchmetrics objects for the epoch-pooled AUROCs, keyed by metric name.

        ``auroc_pixel`` / ``auroc_image`` accumulate across the epoch (reset only at the
        ``(stage, epoch)`` boundary), so the trainer logs these objects with ``on_epoch=True``
        and Lightning computes the single pooled AUROC and resets at epoch end — exact and
        batch-size-invariant, unlike the per-batch running values emitted in ``forward``.
        Returns an empty mapping until the first batch has been seen, so nothing is logged
        for a run that never produced scores.
        """
        if self._last_key is None:
            return {}
        return {"auroc_pixel": self.pixel_auroc, "auroc_image": self.image_auroc}
