"""Tests for DinomalyDetector rectangular (non-square) image_size support.

Covers:
- _to_hw() helper: int → square, tuple → (h,w), list → (h,w), bad input raises
- Square pipeline: _rect_patch_applied stays False (anomalib code path untouched)
- Non-square pipeline: _rect_patch_applied set to True on the model
- Forward output shape is correct for a rectangular input (434 × 1036)
- Backward compat: int image_size produces same output shape as (n, n) tuple
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from cuvis_ai_dinomaly.node.dinomaly_detector import DinomalyDetector

# ---------------------------------------------------------------------------
# Fake DinomalyModel whose encoder exposes a real Conv2d patch-embed
# (needed because DinomalyDetector reads kernel_size from it for patch_size)
# ---------------------------------------------------------------------------


class _FakePatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # kernel_size=14 matches the real ViT-B/14 patch size
        self.proj = nn.Conv2d(3, 8, kernel_size=14, stride=14, bias=False)
        self.in_chans = 3


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _FakePatchEmbed()


class _FakeDinomalyModel(nn.Module):
    """Stand-in: encoder exposes patch_embed.proj (for patch_size read + inflation),
    forward returns a plausible anomaly_map and pred_score."""

    def __init__(self, **_: object) -> None:
        super().__init__()
        self.encoder = _FakeEncoder()
        self.bottleneck = nn.Linear(1, 1, bias=False)
        self.decoder = nn.Linear(1, 1, bias=False)

    def forward(
        self, x: torch.Tensor, global_step: int | None = None
    ) -> torch.Tensor | SimpleNamespace:
        if global_step is not None:
            return x.mean().reshape(())
        # Return an anomaly map at 1/14 of the input spatial resolution
        # (matching the real model's patch-level output before upsampling)
        H, W = x.shape[2], x.shape[3]
        amap = torch.ones(x.shape[0], H // 14, W // 14, device=x.device, dtype=x.dtype)
        score = torch.full((x.shape[0],), 0.42, device=x.device, dtype=x.dtype)
        return SimpleNamespace(anomaly_map=amap, pred_score=score)


def _make_detector(**kw) -> DinomalyDetector:
    with patch(
        "anomalib.models.image.dinomaly.torch_model.DinomalyModel",
        side_effect=lambda **k: _FakeDinomalyModel(**k),
    ):
        return DinomalyDetector(encoder_name="fake", **kw)


# ---------------------------------------------------------------------------
# _to_hw: arg parsing
# ---------------------------------------------------------------------------


def test_image_size_int_stored_as_square_tuple() -> None:
    det = _make_detector(image_size=448, crop_size=392)
    assert det.image_size == (448, 448)
    assert det.crop_size == (392, 392)


def test_image_size_tuple_stored_as_given() -> None:
    det = _make_detector(image_size=(434, 1036), crop_size=(420, 1022))
    assert det.image_size == (434, 1036)
    assert det.crop_size == (420, 1022)


def test_image_size_list_stored_as_tuple() -> None:
    det = _make_detector(image_size=[434, 1036], crop_size=[420, 1022])
    assert det.image_size == (434, 1036)
    assert det.crop_size == (420, 1022)


def test_image_size_bad_input_raises() -> None:
    with pytest.raises(ValueError, match="image_size"):
        _make_detector(image_size="448x448")


def test_crop_size_bad_input_raises() -> None:
    with pytest.raises(ValueError, match="crop_size"):
        _make_detector(image_size=448, crop_size=(1, 2, 3))


# ---------------------------------------------------------------------------
# Rectangular patch: applied only when h != w
# ---------------------------------------------------------------------------


def test_square_pipeline_does_not_apply_rect_patch() -> None:
    """Square image_size must NOT touch the anomalib model — backward compat."""
    det = _make_detector(image_size=448)
    assert not getattr(det.dinomaly_model, "_rect_patch_applied", False), (
        "Square pipeline must not set _rect_patch_applied on the model"
    )


def test_rectangular_pipeline_applies_rect_patch() -> None:
    """Non-square image_size must install the rectangular patch on the model."""
    det = _make_detector(image_size=(434, 1036))
    assert getattr(det.dinomaly_model, "_rect_patch_applied", False), (
        "Rectangular pipeline must set _rect_patch_applied=True on the model"
    )


def test_rect_patch_is_idempotent() -> None:
    """Calling patch_dinomaly_model_for_rectangular_input twice must be a no-op."""
    from cuvis_ai_dinomaly.node._rectangular_input_patch import (
        patch_dinomaly_model_for_rectangular_input,
    )

    det = _make_detector(image_size=(434, 1036))
    # Already patched by constructor; calling again must not raise or re-bind
    original_method = det.dinomaly_model.get_encoder_decoder_outputs
    patch_dinomaly_model_for_rectangular_input(det.dinomaly_model, patch_size=14)
    assert det.dinomaly_model.get_encoder_decoder_outputs is original_method, (
        "Second call must not replace the already-bound method"
    )


# ---------------------------------------------------------------------------
# anomalib version guard: the reimpl is a verbatim copy of anomalib internals,
# so it must refuse to run against an unverified anomalib (would silently diverge).
# ---------------------------------------------------------------------------


def test_version_guard_passes_on_verified_anomalib() -> None:
    """The installed (pinned 2.1.0) anomalib is in the verified set → no raise."""
    import importlib.metadata

    from cuvis_ai_dinomaly.node import _rectangular_input_patch as rp

    assert importlib.metadata.version("anomalib") in rp._VERIFIED_ANOMALIB_VERSIONS
    rp._assert_anomalib_version_verified()  # must not raise


def test_version_guard_raises_on_unverified_anomalib() -> None:
    """An anomalib version outside the verified set must raise (not silently run the
    stale verbatim reimpl)."""
    from cuvis_ai_dinomaly.node import _rectangular_input_patch as rp

    with patch.object(rp.importlib.metadata, "version", return_value="2.99.0"):
        with pytest.raises(RuntimeError, match="rectangular-input patch"):
            rp._assert_anomalib_version_verified()


def test_version_guard_warns_when_version_unknown() -> None:
    """If anomalib has no dist metadata (source/editable), warn + proceed, don't break."""
    from cuvis_ai_dinomaly.node import _rectangular_input_patch as rp

    with patch.object(
        rp.importlib.metadata,
        "version",
        side_effect=rp.importlib.metadata.PackageNotFoundError("anomalib"),
    ):
        rp._assert_anomalib_version_verified()  # must not raise


def test_rect_patch_blocks_unverified_anomalib_at_install() -> None:
    """The guard fires through the public entrypoint on a fresh (unpatched) model."""
    from cuvis_ai_dinomaly.node import _rectangular_input_patch as rp

    model = _FakeDinomalyModel()
    with patch.object(rp.importlib.metadata, "version", return_value="2.99.0"):
        with pytest.raises(RuntimeError, match="silently diverge"):
            rp.patch_dinomaly_model_for_rectangular_input(model, patch_size=14)
    assert not getattr(model, "_rect_patch_applied", False), (
        "model must not be marked patched when the version guard rejects it"
    )


# ---------------------------------------------------------------------------
# Forward output shapes
# ---------------------------------------------------------------------------


def _ctx_inference():
    from cuvis_ai_schemas.enums import ExecutionStage
    from cuvis_ai_schemas.execution import Context

    return Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)


def test_square_forward_output_shape() -> None:
    """Scores output shape should be [B, H, W, 1] matching the original input H×W."""
    det = _make_detector(image_size=448, crop_size=392)
    # Use 28×28 as a tiny stand-in (multiple of 14) so the fake model stays fast
    x = torch.rand(1, 28, 28, 3)
    out = det(x, context=_ctx_inference())
    assert out["scores"].shape == (1, 28, 28, 1), f"unexpected scores shape {out['scores'].shape}"
    assert out["anomaly_score"].shape == (1,)
    assert out["scores"].dtype == torch.float32


def test_rectangular_forward_output_shape() -> None:
    """Scores shape must match the *input* H×W, not the training image_size."""
    det = _make_detector(image_size=(434, 1036), crop_size=(420, 1022))
    # Use (28, 56) = 2× height, 4× width patch grid — proportionally rectangular
    x = torch.rand(1, 28, 56, 3)
    out = det(x, context=_ctx_inference())
    assert out["scores"].shape == (1, 28, 56, 1), f"unexpected scores shape {out['scores'].shape}"
    assert out["anomaly_score"].shape == (1,)
    assert out["scores"].dtype == torch.float32


def test_int_and_tuple_square_give_same_output_shape() -> None:
    """image_size=448 and image_size=(448, 448) must produce identical output shapes."""
    det_int = _make_detector(image_size=448, crop_size=392)
    det_tup = _make_detector(image_size=(448, 448), crop_size=(392, 392))
    x = torch.rand(1, 28, 28, 3)
    out_int = det_int(x, context=_ctx_inference())
    out_tup = det_tup(x, context=_ctx_inference())
    assert out_int["scores"].shape == out_tup["scores"].shape
