"""Shared scaffolding for streaming, histogram-binned AUROC metric nodes (val/test only).

Both :class:`~cuvis_ai_dinomaly.node.auroc_metrics.AnomalyAUROCMetrics` (pixel + image) and
:class:`~cuvis_ai_dinomaly.node.per_class_auroc.PerClassAnomalyAUROC` *compose* torchmetrics
``BinaryAUROC`` accumulators (histogram ``thresholds`` -> O(thresholds) state) rather than
reimplementing AUROC. This base owns the shared bits: the ``thresholds`` node param, the
``(stage, epoch)`` reset bookkeeping, and the sigmoid+flatten transform that maps a raw anomaly
map onto the binned metric's ``[0, 1]`` threshold grid (AUROC is rank-invariant under the
monotonic sigmoid, so the value is unchanged). Subclasses build their own ``BinaryAUROC``
accumulators, implement :meth:`_reset_state`, and define ``forward``.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

Tensor = torch.Tensor


class _StreamingBinnedAUROC(Node):
    """Base for streaming binned-AUROC metric nodes (defaults to VAL/TEST execution)."""

    def __init__(self, thresholds: int = 200, **kwargs: Any) -> None:
        self.thresholds = thresholds
        name, execution_stages = Node.consume_base_kwargs(
            kwargs, {ExecutionStage.VAL, ExecutionStage.TEST}
        )
        super().__init__(
            name=name, execution_stages=execution_stages, thresholds=thresholds, **kwargs
        )
        self._last_key: tuple[ExecutionStage, int] | None = None

    def _reset_state(self) -> None:
        """Reset every streaming accumulator (plus any per-run bookkeeping). Subclass hook."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset accumulators (called by the Predictor before a run, and by tests)."""
        self._reset_state()
        self._last_key = None

    def _reset_on_epoch_boundary(self, context: Context) -> None:
        """Reset once per ``(stage, epoch)`` so each epoch accumulates fresh."""
        key = (context.stage, context.epoch)
        if self._last_key != key:
            self._reset_state()
            self._last_key = key

    @staticmethod
    def _binned_preds(scores: Tensor) -> Tensor:
        """Flatten a raw anomaly map and sigmoid it onto ``[0, 1]`` for the binned metric."""
        return torch.sigmoid(scores.flatten().float())
