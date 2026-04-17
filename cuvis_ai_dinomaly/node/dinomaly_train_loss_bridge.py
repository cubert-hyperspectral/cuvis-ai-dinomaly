"""Loss node that exposes Dinomaly training loss to GradientTrainer."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai.node.losses import LossNode
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class DinomalyTrainLossBridge(LossNode):
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
