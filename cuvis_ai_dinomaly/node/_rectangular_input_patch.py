"""Monkey-patch anomalib's `DinomalyModel` to accept rectangular (non-square) inputs.

Why this is needed
------------------
anomalib's `DinomalyModel.get_encoder_decoder_outputs` (anomalib 2.1.0,
`anomalib/models/image/dinomaly/torch_model.py`, lines 219 + 396) assumes the
patch grid is **square**:

    side = int(math.sqrt(encoder_features[0].shape[1] - 1 - num_register_tokens))
    ...
    f.permute(0, 2, 1).reshape([B, -1, side, side])

This breaks for any input whose H ≠ W (very common in hyperspectral — our bedding
cubes are 1800×4300, aspect 2.39). The DINOv2 encoder itself handles rectangular
inputs natively via positional-embedding interpolation; the bug is purely in the
post-encoder reshape.

This patch replaces the two affected methods with versions that compute
(H_p, W_p) from the input image shape (`x.shape[2:] / patch_size`) and reshape
to that actual grid. For square inputs the behavior is bit-for-bit identical
(H_p == W_p == side); for rectangular inputs it now works.

Usage
-----
Call once after `DinomalyModel.__init__` finishes (which is what we do in
`DinomalyDetector.__init__` when `image_size` is a tuple).
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

import torch
from loguru import logger

# anomalib versions whose ``get_encoder_decoder_outputs`` /
# ``_process_features_for_spatial_output`` internals the reimplementations below
# reproduce **verbatim** (see module docstring). The square path delegates to
# anomalib; this rectangular path is a frozen copy, so it can silently diverge if
# anomalib changes those two methods. When the installed anomalib is not in this
# set, re-verify the reimplementation against the new
# ``anomalib/models/image/dinomaly/torch_model.py`` and then add the version here.
_VERIFIED_ANOMALIB_VERSIONS = frozenset({"2.1.0"})


def _assert_anomalib_version_verified() -> None:
    """Fail loudly if the rectangular reimpl hasn't been verified against the
    installed anomalib (its internals are copied verbatim and can drift silently)."""
    try:
        installed = importlib.metadata.version("anomalib")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        # anomalib present but without dist metadata (vendored / source / editable):
        # we cannot verify the version, so warn rather than break a legit install.
        logger.warning(
            "Could not determine the installed anomalib version; the rectangular-input "
            "patch reimplements anomalib internals verbatim and is verified only against "
            "{}. Proceeding unverified.",
            sorted(_VERIFIED_ANOMALIB_VERSIONS),
        )
        return
    if installed not in _VERIFIED_ANOMALIB_VERSIONS:
        raise RuntimeError(
            f"cuvis-ai-dinomaly's rectangular-input patch reimplements "
            f"DinomalyModel.get_encoder_decoder_outputs and "
            f"_process_features_for_spatial_output verbatim (see "
            f"_rectangular_input_patch.py docstring), but those were copied from anomalib "
            f"{sorted(_VERIFIED_ANOMALIB_VERSIONS)} and the installed version is "
            f"{installed!r}. anomalib {installed} may have changed those methods, which "
            f"would make the rectangular path silently diverge from the (delegated) square "
            f"path. Re-verify the reimplementation against anomalib {installed}'s "
            f"anomalib/models/image/dinomaly/torch_model.py, then add {installed!r} to "
            f"_VERIFIED_ANOMALIB_VERSIONS in this file."
        )


def patch_dinomaly_model_for_rectangular_input(model: Any, patch_size: int = 14) -> None:
    """Install non-square-grid-aware versions of two anomalib methods on `model`.

    No-op if already patched. Raises ``RuntimeError`` if the installed anomalib is
    not one this patch has been verified against (the reimpl is a verbatim copy of
    anomalib internals — see :data:`_VERIFIED_ANOMALIB_VERSIONS`).
    """
    if getattr(model, "_rect_patch_applied", False):
        return
    _assert_anomalib_version_verified()

    def _process_features_for_spatial_output_rect(self, features: list, hp: int, wp: int) -> list:
        # Mirror of anomalib's original (lines 388-396) — only the reshape target changes.
        # If remove_class_token is False, the class+register tokens are still on the
        # token axis and must be stripped before the spatial reshape.
        if not self.remove_class_token:
            features = [f[:, 1 + self.encoder.num_register_tokens :, :] for f in features]
        batch_size = features[0].shape[0]
        return [f.permute(0, 2, 1).reshape([batch_size, -1, hp, wp]).contiguous() for f in features]

    def get_encoder_decoder_outputs_rect(self, x: torch.Tensor) -> tuple[list, list]:
        """Reimplementation of anomalib's get_encoder_decoder_outputs with rectangular
        patch-grid support. Computes (H_p, W_p) from x.shape rather than sqrt(N)."""
        H, W = int(x.shape[2]), int(x.shape[3])
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f"rectangular-input-patched DinomalyModel: input H × W must each be a "
                f"multiple of patch_size={patch_size}, got {H} × {W}"
            )
        hp, wp = H // patch_size, W // patch_size

        x = self.encoder.prepare_tokens(x)

        encoder_features: list[torch.Tensor] = []
        decoder_features: list[torch.Tensor] = []

        for i, block in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = block(x)
            else:
                continue
            if i in self.target_layers:
                encoder_features.append(x)

        if self.remove_class_token:
            encoder_features = [
                e[:, 1 + self.encoder.num_register_tokens :, :] for e in encoder_features
            ]

        x = self._fuse_feature(encoder_features)
        for block in self.bottleneck:
            x = block(x)

        for block in self.decoder:
            x = block(x, attn_mask=None)
            decoder_features.append(x)
        decoder_features = decoder_features[::-1]

        en = [
            self._fuse_feature([encoder_features[idx] for idx in idxs])
            for idxs in self.fuse_layer_encoder
        ]
        de = [
            self._fuse_feature([decoder_features[idx] for idx in idxs])
            for idxs in self.fuse_layer_decoder
        ]

        en = _process_features_for_spatial_output_rect(self, en, hp, wp)
        de = _process_features_for_spatial_output_rect(self, de, hp, wp)
        return en, de

    # Bind the replacement method to the instance.
    model.get_encoder_decoder_outputs = get_encoder_decoder_outputs_rect.__get__(model, type(model))
    model._rect_patch_applied = True
    logger.info(
        "DinomalyModel: patched get_encoder_decoder_outputs for rectangular inputs (patch_size={})",
        patch_size,
    )
