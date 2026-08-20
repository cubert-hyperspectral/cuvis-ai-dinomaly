"""Shared scaffolding for streaming, histogram-binned AUROC metric nodes (val/test only).

Both :class:`~cuvis_ai_dinomaly.node.auroc_metrics.AnomalyAUROCMetrics` (pixel + image) and
:class:`~cuvis_ai_dinomaly.node.per_class_auroc.PerClassAnomalyAUROC` *compose* torchmetrics
``BinaryAUROC`` accumulators (histogram ``thresholds`` -> O(thresholds) state) rather than
reimplementing AUROC. This base owns the binned-AUROC bits: the ``thresholds`` node param and
the sigmoid+flatten transform that maps a raw anomaly map onto the binned metric's ``[0, 1]``
threshold grid (AUROC is rank-invariant under the monotonic sigmoid, so the value is
unchanged). The ``(stage, epoch)`` reset bookkeeping lives in
:class:`~cuvis_ai_dinomaly.node._streaming_metric._StreamingMetric`. Subclasses build their
own ``BinaryAUROC`` accumulators, implement :meth:`_reset_state`, and define ``forward``.
"""

from __future__ import annotations

from typing import Any

import torch

from cuvis_ai_dinomaly.node._streaming_metric import _StreamingMetric

Tensor = torch.Tensor


class _StreamingBinnedAUROC(_StreamingMetric):
    """Streaming binned-AUROC base: ``thresholds`` param + sigmoid/flatten transform.

    Inherits the VAL/TEST defaults and ``(stage, epoch)`` reset bookkeeping from
    :class:`~cuvis_ai_dinomaly.node._streaming_metric._StreamingMetric`.
    """

    def __init__(self, thresholds: int = 200, **kwargs: Any) -> None:
        self.thresholds = thresholds
        super().__init__(thresholds=thresholds, **kwargs)

    @staticmethod
    def _binned_preds(scores: Tensor) -> Tensor:
        """Flatten a raw anomaly map and sigmoid it onto ``[0, 1]`` for the binned metric."""
        return torch.sigmoid(scores.flatten().float())
