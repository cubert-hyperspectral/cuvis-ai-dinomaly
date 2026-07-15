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

from typing import Any

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
        super().__init__(thresholds=thresholds, **kwargs)
        # Histogram-based AUROC: O(thresholds) state, accumulated across batches and reset
        # only at the (stage, epoch) boundary, so each forward's value is a running AUROC.
        # The trainer mean-reduces these per epoch (monitoring) — see the module docstring.
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
