"""Dinomaly anomaly detector node (Anomalib DinomalyModel wrapper)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger
from torch import Tensor
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize


class DinomalyDetector(Node):
    """Pixel-level anomaly detection using Anomalib's ``DinomalyModel``.

    Expects RGB (or false-color) images ``[B, H, W, 3]`` from a channel selector.
    Preprocessing matches Anomalib defaults by default: resize, optional center crop,
    ImageNet normalize. Set ``use_center_crop=False`` to use only ``Resize`` + ``Normalize``
    (e.g. when matching a no-crop Anomalib setup).

    Training
    --------
    - Emits ``training_loss`` (scalar) for :class:`DinomalyTrainLossBridge`.
    - Only bottleneck and decoder weights are trainable; the DINOv2 encoder stays frozen.
    - Override :meth:`unfreeze` / :meth:`freeze` preserve encoder freezing.

    Inference
    ---------
    Emits ``scores`` ``[B, H, W, 1]`` and ``anomaly_score`` ``[B]`` from eval forward.
    """

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB image [B, H, W, 3] in float32 (0–1 or 0–255)",
        ),
    }

    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Pixel-wise anomaly scores [B, H, W, 1]",
        ),
        "anomaly_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Image-level anomaly score [B]",
        ),
        "training_loss": PortSpec(
            dtype=torch.float32,
            shape=(),
            description="Scalar Dinomaly training loss (train/val/test stages)",
            optional=True,
        ),
    }

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        bottleneck_dropout: float = 0.2,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
        image_size: int = 448,
        crop_size: int = 392,
        use_center_crop: bool = True,
        **kwargs: Any,
    ) -> None:
        self.encoder_name = encoder_name
        self.bottleneck_dropout = float(bottleneck_dropout)
        self.decoder_depth = int(decoder_depth)
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = bool(remove_class_token)
        self.image_size = int(image_size)
        self.crop_size = int(crop_size)
        self.use_center_crop = bool(use_center_crop)

        super().__init__(
            encoder_name=encoder_name,
            bottleneck_dropout=bottleneck_dropout,
            decoder_depth=decoder_depth,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token,
            image_size=image_size,
            crop_size=crop_size,
            use_center_crop=use_center_crop,
            **kwargs,
        )

        from anomalib.models.image.dinomaly.torch_model import DinomalyModel

        model = DinomalyModel(
            encoder_name=encoder_name,
            bottleneck_dropout=bottleneck_dropout,
            decoder_depth=decoder_depth,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token,
        )
        self.add_module("dinomaly_model", model)
        self._freeze_encoder_unfreeze_head()

        self._preprocess = self._build_preprocess_compose()

    def _build_preprocess_compose(self) -> Compose:
        steps: list = [Resize((self.image_size, self.image_size))]
        if self.use_center_crop:
            steps.append(CenterCrop(self.crop_size))
        steps.append(
            Normalize(mean=list(self.IMAGENET_MEAN), std=list(self.IMAGENET_STD)),
        )
        return Compose(steps)

    def _freeze_encoder_unfreeze_head(self) -> None:
        model = self.dinomaly_model
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        for p in model.bottleneck.parameters():
            p.requires_grad_(True)
        for p in model.decoder.parameters():
            p.requires_grad_(True)

    def unfreeze(self) -> None:
        """Train bottleneck + decoder only; encoder stays frozen."""
        self._frozen = False
        self._freeze_encoder_unfreeze_head()

    def freeze(self) -> None:
        """Freeze all Dinomaly parameters."""
        self._frozen = True
        for p in self.dinomaly_model.parameters():
            p.requires_grad_(False)

    def _rgb_bhwc_to_model_input(self, rgb_image: Tensor) -> Tensor:
        x = rgb_image
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        x = x.clamp(0.0, 1.0)
        x = x.permute(0, 3, 1, 2)
        return self._preprocess(x)

    def forward(
        self,
        rgb_image: Tensor,
        context: Context | None = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        stage = context.stage if context is not None else ExecutionStage.INFERENCE
        gs = int(context.global_step) if context is not None else 0

        x = self._rgb_bhwc_to_model_input(rgb_image)
        h, w = rgb_image.shape[1], rgb_image.shape[2]
        model = self.dinomaly_model

        out: dict[str, Tensor] = {}

        # Dinomaly loss path registers backward hooks; compute it only in TRAIN.
        # VAL/TEST/INFERENCE use eval predictions only.
        need_train_loss = stage == ExecutionStage.TRAIN

        if need_train_loss:
            model.train()
            tl = model(x, global_step=gs)
            out["training_loss"] = tl if tl.dim() == 0 else tl.reshape(())

        # Scores: eval forward (matches Anomalib validation inference path).
        # Always no_grad here: loss backprops only through training_loss branch.
        with torch.no_grad():
            model.eval()
            pred = model(x)

        amap = pred.anomaly_map
        if amap.dim() == 3:
            amap = amap.unsqueeze(1)
        scores = F.interpolate(amap, size=(h, w), mode="bilinear", align_corners=False)
        scores = scores.permute(0, 2, 3, 1)

        out["scores"] = scores
        out["anomaly_score"] = pred.pred_score

        if need_train_loss:
            model.train()

        logger.trace(
            "DinomalyDetector stage={} gs={} scores.shape={} loss={}",
            stage,
            gs,
            out["scores"].shape,
            out.get("training_loss"),
        )

        return out
