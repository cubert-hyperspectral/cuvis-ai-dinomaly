"""Loss node that exposes Dinomaly training loss to GradientTrainer."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class _LossNode(Node):
    """Minimal local LossNode mirroring ``cuvis_ai.node.losses.LossNode``.

    Inlined here so the plugin does not depend on the high-level ``cuvis-ai``
    package, which transitively eagerly imports the proprietary Cuvis SDK
    (sets ``CUVIS`` env var / loads native libs) at import time. We only
    need the constructor's stage-restriction behavior; the upstream class
    is otherwise a marker.
    """

    _category = NodeCategory.LOSS
    _tags = frozenset({NodeTag.TRAINING, NodeTag.DIFFERENTIABLE, NodeTag.TORCH})

    def __init__(self, **kwargs: Any) -> None:
        assert "execution_stages" not in kwargs, (
            "Loss nodes can only execute in train, val, and test stages."
        )
        super().__init__(
            execution_stages={
                ExecutionStage.TRAIN,
                ExecutionStage.VAL,
                ExecutionStage.TEST,
            },
            **kwargs,
        )


class DinomalyTrainLossBridge(_LossNode):
    """Passes through the scalar reconstruction loss from :class:`DinomalyDetector`.

    Connect ``dinomaly_detector.outputs.training_loss`` to ``raw_loss``.
    """

    INPUT_SPECS = {
        "raw_loss": PortSpec(
            dtype=torch.float32,
            shape=(),
            description="Scalar training loss from DinomalyDetector",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "loss": PortSpec(dtype=torch.float32, shape=(), description="Weighted loss for backprop"),
    }

    def __init__(self, weight: float = 1.0, **kwargs: Any) -> None:
        self.weight = float(weight)
        super().__init__(weight=self.weight, **kwargs)

    def forward(self, raw_loss: Tensor | None = None, **_: Any) -> dict[str, Tensor]:
        if raw_loss is None:
            return {"loss": torch.tensor(0.0, dtype=torch.float32)}
        if raw_loss.dim() != 0:
            raw_loss = raw_loss.reshape(())
        return {"loss": raw_loss * self.weight}
