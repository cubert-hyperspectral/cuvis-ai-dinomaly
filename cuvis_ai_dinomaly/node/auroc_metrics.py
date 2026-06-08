"""Epoch-level pixel-AUROC and image-AUROC for Dinomaly.

Ported 1:1 from the equivalent Dinomaly2 implementation. The correct long-term home
is a generic ``AnomalyAUROCMetrics`` node in ``cuvis-ai`` (metrics.py) with a matching
epoch-start/end hook added to ``cuvis-ai-core``'s ``GradientTrainer``.

Design
------
``AnomalyAUROCMetrics`` is a plain :class:`~cuvis_ai_core.node.node.Node` that
accumulates raw predictions on CPU during forward passes (val/test only) and returns
an empty metrics list each batch. :class:`AUROCEpochEndCallback` is a Lightning
:class:`~lightning.pytorch.callbacks.Callback` that:

- calls ``node.reset()`` on epoch start (val and test)
- calls ``node.compute_auroc()`` on epoch end and logs via ``pl_module.log()``

Both classes must be wired together in the train-common script. See
``cuvis-ai-cookbook/examples/bedding_dinomaly/train_bedding_all6.py`` for the
expected wiring pattern.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from lightning.pytorch.callbacks import Callback
from loguru import logger

Tensor = torch.Tensor


class AnomalyAUROCMetrics(Node):
    """Accumulate pixel/image scores across an epoch for AUROC computation.

    Does NOT emit any ``Metric`` objects per batch — AUROC is only meaningful at the
    epoch level. :class:`AUROCEpochEndCallback` triggers the actual compute.

    Input ports
    -----------
    scores : ``[B, H, W, 1] float32`` — raw anomaly map (not thresholded)
    targets : ``[B, H, W, 1] bool``   — ground-truth pixel masks
    anomaly_score : ``[B] float32``    — per-image score (top-k mean of amap)

    Output ports
    ------------
    metrics : ``list[Metric]`` — always empty; logging happens via the callback.
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
        "metrics": PortSpec(dtype=list, shape=(), description="Always empty list"),
    }

    def __init__(self, **kwargs: Any) -> None:
        name, execution_stages = Node.consume_base_kwargs(
            kwargs, {ExecutionStage.VAL, ExecutionStage.TEST}
        )
        super().__init__(name=name, execution_stages=execution_stages, **kwargs)
        self._pixel_preds: list[Tensor] = []
        self._pixel_targets: list[Tensor] = []
        self._image_preds: list[Tensor] = []
        self._image_targets: list[Tensor] = []

    def reset(self) -> None:
        self._pixel_preds.clear()
        self._pixel_targets.clear()
        self._image_preds.clear()
        self._image_targets.clear()

    def compute_auroc(self) -> dict[str, float]:
        """Compute epoch-level AUROC from accumulated predictions.

        Returns an empty dict if no data was accumulated or if all targets are the
        same class (AUROC is undefined in that case).
        """
        if not self._pixel_preds:
            return {}

        try:
            from torchmetrics.functional.classification import binary_auroc
        except ImportError:
            logger.warning("torchmetrics not available — skipping AUROC computation")
            return {}

        results: dict[str, float] = {}

        # Pixel AUROC.
        all_pixel_preds = torch.cat(self._pixel_preds)
        all_pixel_tgts = torch.cat(self._pixel_targets)
        if all_pixel_tgts.long().sum() > 0 and (~all_pixel_tgts).long().sum() > 0:
            results["auroc_pixel"] = float(binary_auroc(all_pixel_preds, all_pixel_tgts.long()))
        else:
            logger.warning("Pixel AUROC skipped: all targets are the same class")

        # Image AUROC.
        all_image_preds = torch.cat(self._image_preds)
        all_image_tgts = torch.cat(self._image_targets)
        if all_image_tgts.sum() > 0 and (all_image_tgts == 0).sum() > 0:
            results["auroc_image"] = float(binary_auroc(all_image_preds, all_image_tgts))
        else:
            logger.warning("Image AUROC skipped: all images are the same class")

        return results

    def forward(
        self,
        scores: Tensor,
        targets: Tensor,
        anomaly_score: Tensor,
        context: Context,  # noqa: ARG002
    ) -> dict[str, Any]:
        # Accumulate pixel-level predictions on CPU so the GPU doesn't keep huge
        # 1800×4300 score maps live until epoch end.
        self._pixel_preds.append(scores.squeeze(-1).flatten().float().detach().cpu())
        self._pixel_targets.append(targets.squeeze(-1).flatten().bool().detach().cpu())

        # Image label: anomalous if any pixel in the GT mask is positive.
        img_labels = targets.squeeze(-1).flatten(1).any(dim=1).long().detach().cpu()
        self._image_preds.append(anomaly_score.float().detach().cpu())
        self._image_targets.append(img_labels)

        return {"metrics": []}


class AUROCEpochEndCallback(Callback):
    """Reset accumulator at epoch start; compute and log AUROC at epoch end.

    Pass an instance of this to ``GradientTrainer(callbacks=[...])`` alongside
    :class:`AnomalyAUROCMetrics` wired into the pipeline graph.
    """

    def __init__(
        self, auroc_node: AnomalyAUROCMetrics, node_log_prefix: str = "metrics_auroc"
    ) -> None:
        super().__init__()
        self._node = auroc_node
        self._prefix = node_log_prefix

    # ------------------------------------------------------------------ val ----
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._node.reset()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = self._node.compute_auroc()
        for name, value in metrics.items():
            pl_module.log(f"{self._prefix}/{name}", value, prog_bar=True, on_epoch=True)
            logger.info("Val epoch AUROC — {}/{}: {:.4f}", self._prefix, name, value)
        # Defragment the allocator before returning to training: val forward passes
        # (eval-mode anomaly map + Gaussian smoothing) leave many small reserved
        # segments that can prevent training-step allocations.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ test ---
    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._node.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = self._node.compute_auroc()
        for name, value in metrics.items():
            pl_module.log(f"{self._prefix}/{name}", value, prog_bar=True, on_epoch=True)
            logger.info("Test epoch AUROC — {}/{}: {:.4f}", self._prefix, name, value)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
