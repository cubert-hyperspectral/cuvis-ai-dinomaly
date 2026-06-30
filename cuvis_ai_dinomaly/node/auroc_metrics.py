"""Streaming pixel-AUROC + image-AUROC for Dinomaly (validation / test).

Mirrors the :class:`cuvis_ai.node.metrics.AnomalyDetectionMetrics` pattern: a
``torchmetrics`` ``BinaryAUROC`` with histogram ``thresholds`` (so per-epoch state is
O(thresholds), not the couple-GB-per-epoch CPU concat of every pixel) is accumulated
via ``update()`` across batches and reset on the ``(stage, epoch)`` boundary. Each
forward emits the *running* AUROC as a :class:`~cuvis_ai_schemas.execution.Metric`. No
bespoke Lightning callback is needed.

This is a **training-time monitoring** metric, not the authoritative score. The trainer
logs each ``Metric.value`` as a float per batch and Lightning reduces per-epoch with its
``on_epoch`` default (mean), so the reported epoch scalar is the *mean of the per-batch
running AUROCs* — an approximation of, not equal to, the exact pooled AUROC. Core 0.10
exposes no epoch-end hook and ``Context`` has no last-batch flag, so the node cannot force
a single exact compute through this channel. The authoritative whole-dataset AUROC is
computed separately (sklearn ``roc_auc_score`` over all pooled frames in the bedding eval
script); that is what the published metrics use.

Scores are passed through ``sigmoid`` before the binned metric so the thresholds span
``[0, 1]``; AUROC is rank-invariant under a monotonic transform, so the value is
unchanged.

NOTE (deferred): the long-term home is upstream next to ``AnomalyDetectionMetrics`` in
``cuvis-ai`` ``metrics.py``. Moving it is deferred because it would break the import
path of already-saved pipelines (incl. the published HF model) and needs a cuvis-ai
release + a pipeline re-point — tracked alongside the selector retirement.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context, Metric
from cuvis_ai_schemas.pipeline import PortSpec
from torchmetrics.classification import BinaryAUROC

Tensor = torch.Tensor


class AnomalyAUROCMetrics(Node):
    """Streaming pixel/image AUROC via torchmetrics (val/test only).

    Input ports
    -----------
    scores : ``[B, H, W, 1] float32`` — raw anomaly map (not thresholded)
    targets : ``[B, H, W, 1] bool``   — ground-truth pixel masks
    anomaly_score : ``[B] float32``    — per-image score (top-k mean of amap)

    Output ports
    ------------
    metrics : ``list[Metric]`` — running ``auroc_pixel`` / ``auroc_image`` (monitoring;
        the trainer mean-reduces these per epoch — see the module docstring).
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.ANOMALY})

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
        self.thresholds = thresholds
        name, execution_stages = Node.consume_base_kwargs(
            kwargs, {ExecutionStage.VAL, ExecutionStage.TEST}
        )
        super().__init__(
            name=name, execution_stages=execution_stages, thresholds=thresholds, **kwargs
        )
        # Histogram-based AUROC: O(thresholds) state, accumulated across batches and reset
        # only at the (stage, epoch) boundary, so each forward's value is a running AUROC.
        # The trainer mean-reduces these per epoch (monitoring) — see the module docstring.
        self.pixel_auroc = BinaryAUROC(thresholds=thresholds)
        self.image_auroc = BinaryAUROC(thresholds=thresholds)
        self._last_key: tuple[ExecutionStage, int] | None = None

    def reset(self) -> None:
        """Reset both running accumulators (kept for explicit external use/tests)."""
        self.pixel_auroc.reset()
        self.image_auroc.reset()
        self._last_key = None

    def forward(
        self,
        scores: Tensor,
        targets: Tensor,
        anomaly_score: Tensor,
        context: Context,
    ) -> dict[str, Any]:
        # Reset on the (stage, epoch) boundary so each epoch accumulates fresh.
        key = (context.stage, context.epoch)
        if self._last_key != key:
            self.pixel_auroc.reset()
            self.image_auroc.reset()
            self._last_key = key

        # Pixel-level: sigmoid -> [0, 1] for the binned metric (AUROC is rank-invariant).
        pix_preds = torch.sigmoid(scores.squeeze(-1).flatten().float())
        pix_tgts = targets.squeeze(-1).flatten().long()
        self.pixel_auroc.update(pix_preds, pix_tgts)

        # Image-level: per-image score vs "any GT pixel positive" label.
        img_preds = torch.sigmoid(anomaly_score.float().flatten())
        img_tgts = targets.squeeze(-1).flatten(1).any(dim=1).long()
        self.image_auroc.update(img_preds, img_tgts)

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
