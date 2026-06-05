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

    Expects channel-stacked images ``[B, H, W, input_channels]`` from a channel selector.
    ``input_channels`` defaults to 3 (RGB / false-color, fully backward compatible).
    For multi-channel hyperspectral input (e.g. ``input_channels=6`` for VIS+SWIR),
    the DINOv2 patch-embed conv is inflated from 3→N input channels via duplicate+halve
    (see ``_patch_embed_inflation.py``), and the inflated patch-embed conv becomes
    trainable while the rest of the encoder stays frozen.

    Channel-order contract for ``input_channels > 3``: the caller MUST supply input
    channels grouped as ``[c0, c1, c2, c0', c1', c2', ...]`` (descending λ within each
    triplet, semantically-paired triplets stacked). This matches the inflated conv's
    weight layout so each output filter sees consistent per-slot input statistics at init.

    Preprocessing matches Anomalib defaults by default: resize, optional center crop,
    ImageNet normalize. For ``input_channels > 3`` the ImageNet mean/std vectors are
    tiled (e.g. ``[R,G,B,R,G,B]`` for 6ch) — a neutral choice that pairs with the
    duplicate-and-halve inflation; the patch-embed will be fine-tuned from epoch 0.

    Training
    --------
    - Emits ``training_loss`` (scalar) for :class:`DinomalyTrainLossBridge`.
    - 3-ch path: only bottleneck and decoder are trainable; DINOv2 encoder stays frozen.
    - >3-ch path: same as above PLUS the inflated patch-embed conv is trainable.

    Inference
    ---------
    Emits ``scores`` ``[B, H, W, 1]`` and ``anomaly_score`` ``[B]`` from eval forward.
    """

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Channel-stacked image [B, H, W, C] in float32 (0–1 or 0–255). "
                        "C must equal the detector's input_channels (default 3).",
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

    # Base (3-channel) ImageNet statistics. For input_channels>3 these are tiled
    # at __init__ time into instance attributes of the right length.
    _IMAGENET_MEAN_3 = (0.485, 0.456, 0.406)
    _IMAGENET_STD_3 = (0.229, 0.224, 0.225)

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        bottleneck_dropout: float = 0.2,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
        image_size: int | tuple[int, int] | list[int] = 448,
        crop_size: int | tuple[int, int] | list[int] = 392,
        use_center_crop: bool = True,
        input_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        self.encoder_name = encoder_name
        self.bottleneck_dropout = float(bottleneck_dropout)
        self.decoder_depth = int(decoder_depth)
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = bool(remove_class_token)
        # Accept int (square) or (h, w) tuple/list (aspect-preserving). Internally
        # store as (h, w) tuple so downstream code is shape-agnostic. A bare int
        # behaves exactly as before (backward compatible with all saved pipelines).
        def _to_hw(x, name: str) -> tuple[int, int]:
            if isinstance(x, int):
                return (int(x), int(x))
            if isinstance(x, (tuple, list)) and len(x) == 2:
                return (int(x[0]), int(x[1]))
            raise ValueError(f"{name} must be int or (h, w) tuple/list, got {x!r}")
        self.image_size = _to_hw(image_size, "image_size")
        self.crop_size = _to_hw(crop_size, "crop_size")
        self.use_center_crop = bool(use_center_crop)
        self.input_channels = int(input_channels)
        if self.input_channels <= 0 or self.input_channels % 3 != 0:
            raise ValueError(
                f"DinomalyDetector: input_channels must be a positive multiple of 3 "
                f"(got {self.input_channels}). The patch-embed inflation needs the new "
                f"channel count to be an integer multiple of the pretrained 3."
            )
        # Instance-level ImageNet stats: tile the 3-vectors to length input_channels.
        # E.g. for 6ch: [R,G,B,R,G,B] — pairs with the duplicate-and-halve patch-embed
        # surgery so each output filter sees matched per-slot input statistics at init.
        factor = self.input_channels // 3
        self.IMAGENET_MEAN = self._IMAGENET_MEAN_3 * factor
        self.IMAGENET_STD = self._IMAGENET_STD_3 * factor

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
            input_channels=input_channels,
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
        # Rectangular-input support: anomalib's DinomalyModel hard-codes a square
        # patch-grid reshape (sqrt(N_tokens) × sqrt(N_tokens)). Only patch when
        # we're actually using a rectangular input — square inputs keep the
        # original code path untouched for full backward compat.
        if self.image_size[0] != self.image_size[1]:
            from cuvis_ai_dinomaly.node._rectangular_input_patch import (
                patch_dinomaly_model_for_rectangular_input,
            )
            # Infer patch_size from the encoder's patch_embed.proj (kernel = stride = patch)
            patch_size = int(model.encoder.patch_embed.proj.kernel_size[0])
            patch_dinomaly_model_for_rectangular_input(model, patch_size=patch_size)
        # Patch-embed inflation for multi-channel input. Done AFTER the pretrained
        # encoder has loaded so we surgically replace the 3-ch proj in place.
        if self.input_channels != 3:
            from cuvis_ai_dinomaly.node._patch_embed_inflation import (
                inflate_conv2d_input_channels,
            )
            old_proj = model.encoder.patch_embed.proj
            new_proj = inflate_conv2d_input_channels(old_proj, self.input_channels)
            model.encoder.patch_embed.proj = new_proj
            # Some PatchEmbed implementations cache in_chans for downstream sanity checks.
            if hasattr(model.encoder.patch_embed, "in_chans"):
                model.encoder.patch_embed.in_chans = self.input_channels
            logger.info(
                "DinomalyDetector: inflated patch-embed Conv2d from {} → {} input channels",
                old_proj.in_channels, self.input_channels,
            )
        self._freeze_encoder_unfreeze_head()

        self._preprocess = self._build_preprocess_compose()

    def _build_preprocess_compose(self) -> Compose:
        # self.image_size and self.crop_size are always (h, w) tuples (see __init__).
        steps: list = [Resize(self.image_size)]
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
        # When the patch-embed was inflated, its SWIR-slot weights are duplicates of
        # the VIS-slot weights — they need to be trainable to differentiate. The rest
        # of the encoder (transformer blocks) remains frozen.
        if self.input_channels != 3:
            for p in model.encoder.patch_embed.proj.parameters():
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
        max_val = float(x.max())
        # Scale to [0, 1] robustly by dividing by the per-cube max whenever it
        # exceeds 1.0. This handles lentils-style uint16 reflectance (max ~10000),
        # bedding-style float reflectance with specular highlights (max ~38000),
        # and legacy uint8 RGB (max ~255) uniformly — for fully-saturated uint8
        # input this is mathematically identical to the old "/ 255" path, and
        # for under-saturated uint8 input the slightly higher contrast is
        # absorbed by the downstream ImageNet normalize.
        # IMPORTANT: do NOT pre-scale reflectance to [0, 1] in the dataset/
        # converter — store the raw cuvis output (u16-encoded reflectance or
        # equivalent) so this branch fires and per-cube max-scaling stays
        # consistent across the codebase.
        if max_val > 1.0:
            x = x / max_val
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

        # Cast to fp32 to satisfy OUTPUT_SPECS regardless of autocast / mixed-precision.
        # Lightning's precision="16-mixed" makes the model return fp16 tensors, which the
        # pipeline runtime port validator would reject against the float32 declaration.
        out["scores"] = scores.to(torch.float32)
        out["anomaly_score"] = pred.pred_score.to(torch.float32)

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
