"""Label-free validation monitoring: running mean anomaly score on normal frames."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context, Metric
from cuvis_ai_schemas.pipeline import PortSpec

Tensor = torch.Tensor


class ValNormalAnomalyMean(Node):
    """Running mean of the image-level ``anomaly_score`` per ``(stage, epoch)``.

    For unsupervised (normals-only) training the val split has no masks, so
    IoU/AUROC metrics are degenerate and Dinomaly's reconstruction loss is
    TRAIN-only. On held-out NORMAL frames the mean anomaly score should fall
    as the model learns normality; a sustained rise while the train loss falls
    is an overfitting signal. VAL/TEST stages only; resets on the
    ``(stage, epoch)`` boundary (mirrors the
    :class:`~cuvis_ai_dinomaly.node.auroc_metrics.AnomalyAUROCMetrics` pattern).

    Input ports
    -----------
    anomaly_score : ``[B] float32`` — per-image score (top-k mean of amap)

    Output ports
    ------------
    metrics : ``list[Metric]`` — running ``mean_anomaly_normal`` for the
        current ``(stage, epoch)``.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.ANOMALY})

    INPUT_SPECS = {
        "anomaly_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Per-image anomaly score [B]",
        )
    }
    OUTPUT_SPECS = {
        "metrics": PortSpec(
            dtype=list,
            shape=(),
            description="List of Metric objects (running mean_anomaly_normal)",
        )
    }

    def __init__(self, **kwargs: Any) -> None:
        name, execution_stages = Node.consume_base_kwargs(
            kwargs, {ExecutionStage.VAL, ExecutionStage.TEST}
        )
        super().__init__(name=name, execution_stages=execution_stages, **kwargs)
        self._sum = 0.0
        self._n = 0
        self._last_key: tuple[ExecutionStage, int] | None = None

    def reset(self) -> None:
        """Reset accumulators (called by the Predictor before a run, and by tests)."""
        self._sum, self._n, self._last_key = 0.0, 0, None

    def forward(self, anomaly_score: Tensor, context: Context) -> dict[str, Any]:
        # Reset on the (stage, epoch) boundary so each epoch accumulates fresh.
        key = (context.stage, context.epoch)
        if self._last_key != key:
            self._sum, self._n = 0.0, 0
            self._last_key = key
        self._sum += float(anomaly_score.detach().float().sum())
        self._n += int(anomaly_score.numel())
        return {
            "metrics": [
                Metric(
                    name="mean_anomaly_normal",
                    value=self._sum / max(self._n, 1),
                    stage=context.stage,
                    epoch=context.epoch,
                    batch_idx=context.batch_idx,
                )
            ]
        }
