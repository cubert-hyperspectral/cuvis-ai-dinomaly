"""Streaming per-class pixel AUROC (one-vs-background) for Dinomaly (validation / test).

For each non-background class ``c``, a torchmetrics ``BinaryAUROC`` (histogram ``thresholds``,
so per-epoch state is O(thresholds)) accumulates over the pixels labelled ``c`` (positive) or
background (negative); other classes are excluded from that class's metric. This is the streaming,
binned analogue of the pooled sklearn one-vs-background per-class AUROC.

Per-class AUROC is *not* recoverable from a pooled binary AUROC (its state has already discarded
which class each positive pixel came from), so each class needs its own accumulator fed at
``update()`` time. The node therefore composes one ``BinaryAUROC`` per class, reusing the same
binned primitive as :class:`~cuvis_ai_dinomaly.node.auroc_metrics.AnomalyAUROCMetrics` via the
shared :class:`~cuvis_ai_dinomaly.node._binned_auroc._StreamingBinnedAUROC` base rather than
reimplementing AUROC.

Like ``AnomalyAUROCMetrics`` this is a training-time monitoring metric (binned approximation of
the exact pooled AUROC); the notebook reads the whole-split value once via :meth:`compute` after a
single ``Predictor`` pass.
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


class PerClassAnomalyAUROC(_StreamingBinnedAUROC):
    """Streaming one-vs-background pixel AUROC per class (val/test only).

    Input ports
    -----------
    scores : ``[B, H, W, 1] float32`` — raw anomaly map (not thresholded)
    class_mask : ``[B, H, W, 1] int32`` — multi-class ground truth (``background_id`` = normal)

    Output ports
    ------------
    metrics : ``list[Metric]`` — running ``auroc_pixel_<class>`` for classes seen so far
        (monitoring; the authoritative whole-split value is :meth:`compute`).
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.ANOMALY})

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Raw anomaly map [B, H, W, 1]",
        ),
        "class_mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1, 1),
            description="Multi-class ground-truth mask [B, H, W, 1] (background_id = normal)",
        ),
    }
    OUTPUT_SPECS = {
        "metrics": PortSpec(
            dtype=list,
            shape=(),
            description="List of Metric objects (running per-class AUROC)",
        ),
    }

    def __init__(
        self,
        class_names: dict[int, str],
        background_id: int = 0,
        thresholds: int = 200,
        **kwargs: Any,
    ) -> None:
        self.class_names = {int(k): str(v) for k, v in class_names.items()}
        self.background_id = int(background_id)
        super().__init__(
            thresholds=thresholds,
            class_names=self.class_names,
            background_id=self.background_id,
            **kwargs,
        )
        # One binned BinaryAUROC per non-background class (ModuleDict keys must be str).
        self._aurocs = torch.nn.ModuleDict(
            {
                str(cid): BinaryAUROC(thresholds=thresholds)
                for cid in self.class_names
                if cid != self.background_id
            }
        )
        # Class ids updated at least once this run (so compute() skips never-seen classes).
        self._seen: set[int] = set()

    def _reset_state(self) -> None:
        for metric in self._aurocs.values():
            metric.reset()
        self._seen.clear()

    def compute(self) -> dict[str, float]:
        """Whole-run per-class pixel AUROC, keyed by class name (never-seen classes omitted)."""
        return {
            self.class_names[cid]: float(self._aurocs[str(cid)].compute())
            for cid in sorted(self._seen)
        }

    def forward(
        self,
        scores: Tensor,
        class_mask: Tensor,
        context: Context,
    ) -> dict[str, Any]:
        # Reset on the (stage, epoch) boundary so each epoch accumulates fresh.
        self._reset_on_epoch_boundary(context)

        preds = self._binned_preds(scores)  # [N] in [0, 1]
        labels = class_mask.squeeze(-1).flatten().long()  # [N] class ids, pixel-aligned with preds
        is_background = labels == self.background_id

        metrics: list[Metric] = []
        for cid_str, metric in self._aurocs.items():
            cid = int(cid_str)
            is_class = labels == cid
            if is_class.any():
                # One-vs-background: positives = this class, negatives = background only.
                keep = is_class | is_background
                metric.update(preds[keep], is_class[keep].long())
                self._seen.add(cid)
            if cid in self._seen:
                metrics.append(
                    Metric(
                        name=f"auroc_pixel_{self.class_names[cid]}",
                        value=float(metric.compute()),
                        stage=context.stage,
                        epoch=context.epoch,
                        batch_idx=context.batch_idx,
                    )
                )
        return {"metrics": metrics}
