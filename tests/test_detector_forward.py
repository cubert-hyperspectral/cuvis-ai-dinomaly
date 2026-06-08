"""Fast DinomalyDetector tests with a patched ``DinomalyModel`` (no weight download)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

from cuvis_ai_dinomaly.node.dinomaly_detector import DinomalyDetector


class _FakeDinomalyModel(nn.Module):
    """Minimal stand-in for Anomalib DinomalyModel (encoder/bottleneck/decoder + forward)."""

    def __init__(self, **_: object) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 1, bias=False)
        self.bottleneck = nn.Linear(1, 1, bias=False)
        self.decoder = nn.Linear(1, 1, bias=False)

    def forward(
        self, x: torch.Tensor, global_step: int | None = None
    ) -> torch.Tensor | SimpleNamespace:
        if global_step is not None:
            base = x.mean()
            return (
                base + self.bottleneck.weight.sum() * 0.0 + self.decoder.weight.sum() * 0.0
            ).reshape(())
        amap = torch.ones(x.shape[0], 28, 28, device=x.device, dtype=x.dtype)
        score = torch.full((x.shape[0],), 0.42, device=x.device, dtype=x.dtype)
        return SimpleNamespace(anomaly_map=amap, pred_score=score)


@pytest.fixture
def patched_dinomaly() -> DinomalyDetector:
    with patch(
        "anomalib.models.image.dinomaly.torch_model.DinomalyModel",
        side_effect=lambda **kw: _FakeDinomalyModel(**kw),
    ):
        return DinomalyDetector(encoder_name="fake")


def test_val_stage_omits_training_loss(patched_dinomaly: DinomalyDetector) -> None:
    det = patched_dinomaly
    ctx = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0, global_step=1)
    out = det(torch.rand(1, 16, 16, 3), context=ctx)
    assert "training_loss" not in out


def test_train_stage_emits_training_loss_and_scores(patched_dinomaly: DinomalyDetector) -> None:
    det = patched_dinomaly
    x = torch.rand(2, 64, 72, 3)
    ctx = Context(stage=ExecutionStage.TRAIN, epoch=0, batch_idx=0, global_step=3)
    out = det(x, context=ctx)
    assert "training_loss" in out
    assert out["training_loss"].dim() == 0
    assert out["scores"].shape == (2, 64, 72, 1)
    assert out["anomaly_score"].shape == (2,)


def test_inference_stage_omits_training_loss(patched_dinomaly: DinomalyDetector) -> None:
    det = patched_dinomaly
    x = torch.rand(1, 32, 40, 3)
    ctx = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)
    out = det(x, context=ctx)
    assert "training_loss" not in out
    assert out["scores"].shape == (1, 32, 40, 1)


def test_forward_without_context_uses_inference_stage(patched_dinomaly: DinomalyDetector) -> None:
    det = patched_dinomaly
    out = det(torch.rand(1, 16, 16, 3))
    assert "training_loss" not in out


def test_uint8_input_scaled_to_float01(patched_dinomaly: DinomalyDetector) -> None:
    det = patched_dinomaly
    x = torch.randint(0, 256, (1, 8, 8, 3), dtype=torch.uint8)
    ctx = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)
    out = det(x, context=ctx)
    assert out["scores"].shape == (1, 8, 8, 1)


def test_float64_input_cast_for_preprocess(patched_dinomaly: DinomalyDetector) -> None:
    det = patched_dinomaly
    x = torch.rand(1, 8, 8, 3, dtype=torch.float64) * 0.5
    ctx = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)
    out = det(x, context=ctx)
    assert torch.isfinite(out["anomaly_score"]).all()


def test_out_of_range_float_input_is_clamped_and_runs(patched_dinomaly: DinomalyDetector) -> None:
    det = patched_dinomaly
    x = torch.randn(1, 12, 14, 3) * 3.0  # includes values below 0 and above 1
    ctx = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)
    out = det(x, context=ctx)
    assert out["scores"].shape == (1, 12, 14, 1)
    assert torch.isfinite(out["scores"]).all()


def test_training_loss_1d_scalar_is_flattened() -> None:
    class _Loss1D(_FakeDinomalyModel):
        def forward(
            self, x: torch.Tensor, global_step: int | None = None
        ) -> torch.Tensor | SimpleNamespace:
            if global_step is not None:
                return torch.tensor([0.5], device=x.device, dtype=x.dtype)
            return super().forward(x, global_step=global_step)

    with patch(
        "anomalib.models.image.dinomaly.torch_model.DinomalyModel",
        side_effect=lambda **kw: _Loss1D(**kw),
    ):
        det = DinomalyDetector(encoder_name="fake")
    ctx = Context(stage=ExecutionStage.TRAIN, epoch=0, batch_idx=0, global_step=0)
    out = det(torch.rand(1, 10, 10, 3), context=ctx)
    assert out["training_loss"].dim() == 0
    assert out["training_loss"].item() == pytest.approx(0.5)


def test_freeze_then_unfreeze_encoder_stays_frozen(patched_dinomaly: DinomalyDetector) -> None:
    det = patched_dinomaly
    det.freeze()
    assert not any(p.requires_grad for p in det.dinomaly_model.parameters())
    det.unfreeze()
    assert not any(p.requires_grad for p in det.dinomaly_model.encoder.parameters())
    assert any(p.requires_grad for p in det.dinomaly_model.bottleneck.parameters())
    assert any(p.requires_grad for p in det.dinomaly_model.decoder.parameters())


def test_outputs_are_float32_when_inputs_are_float16(patched_dinomaly: DinomalyDetector) -> None:
    """fp16 inputs (Lightning's precision='16-mixed' code path) must produce fp32 outputs."""
    det = patched_dinomaly
    x = torch.rand(1, 16, 16, 3, dtype=torch.float16)
    ctx = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)
    out = det(x, context=ctx)
    assert out["scores"].dtype == torch.float32
    assert out["anomaly_score"].dtype == torch.float32


def test_outputs_are_float32_under_autocast(patched_dinomaly: DinomalyDetector) -> None:
    """Outputs must satisfy OUTPUT_SPECS even when the forward runs under autocast."""
    det = patched_dinomaly
    x = torch.rand(1, 16, 16, 3)
    ctx = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = det(x, context=ctx)
    assert out["scores"].dtype == torch.float32
    assert out["anomaly_score"].dtype == torch.float32


def test_eval_forward_uses_bchw_anomaly_map_with_channel_dim(
    patched_dinomaly: DinomalyDetector,
) -> None:
    """Cover interpolate branch when anomaly_map is 4D [B,1,H,W]."""

    class _FourD(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(1, 1, bias=False)
            self.bottleneck = nn.Linear(1, 1, bias=False)
            self.decoder = nn.Linear(1, 1, bias=False)

        def forward(
            self, x: torch.Tensor, global_step: int | None = None
        ) -> torch.Tensor | SimpleNamespace:
            if global_step is not None:
                return x.mean().reshape(())
            amap = torch.ones(x.shape[0], 1, 7, 7, device=x.device, dtype=x.dtype)
            return SimpleNamespace(
                anomaly_map=amap,
                pred_score=torch.zeros(x.shape[0], device=x.device, dtype=x.dtype),
            )

    with patch(
        "anomalib.models.image.dinomaly.torch_model.DinomalyModel",
        side_effect=lambda **kw: _FourD(),
    ):
        det = DinomalyDetector(encoder_name="fake")
    x = torch.rand(1, 20, 24, 3)
    out = det(
        x, context=Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)
    )
    assert out["scores"].shape == (1, 20, 24, 1)


# ---------------------------------------------------------------------------
# Multi-channel input (input_channels > 3) — bedding-all6 pilot
# ---------------------------------------------------------------------------


class _FakePatchEmbed(nn.Module):
    """Real ``Conv2d`` patch-embed proj so the inflation surgery can run unchanged."""

    def __init__(self, in_chans: int = 3, embed_dim: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=14, stride=14, bias=False)
        self.in_chans = in_chans


class _FakeEncoderWithPatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _FakePatchEmbed(in_chans=3, embed_dim=8)


class _FakeDinomalyModelWithPatchEmbed(nn.Module):
    """Fake DinomalyModel whose encoder exposes a real Conv2d at encoder.patch_embed.proj.

    The patch-embed conv is not actually used in ``forward`` (the fake just averages x),
    but DinomalyDetector's inflation surgery needs the path to exist.
    """

    def __init__(self, **_: object) -> None:
        super().__init__()
        self.encoder = _FakeEncoderWithPatchEmbed()
        self.bottleneck = nn.Linear(1, 1, bias=False)
        self.decoder = nn.Linear(1, 1, bias=False)

    def forward(
        self, x: torch.Tensor, global_step: int | None = None
    ) -> torch.Tensor | SimpleNamespace:
        if global_step is not None:
            base = x.mean()
            return (
                base + self.bottleneck.weight.sum() * 0.0 + self.decoder.weight.sum() * 0.0
            ).reshape(())
        amap = torch.ones(x.shape[0], 28, 28, device=x.device, dtype=x.dtype)
        score = torch.full((x.shape[0],), 0.42, device=x.device, dtype=x.dtype)
        return SimpleNamespace(anomaly_map=amap, pred_score=score)


@pytest.fixture
def patched_dinomaly_6ch() -> DinomalyDetector:
    with patch(
        "anomalib.models.image.dinomaly.torch_model.DinomalyModel",
        side_effect=lambda **kw: _FakeDinomalyModelWithPatchEmbed(**kw),
    ):
        return DinomalyDetector(encoder_name="fake", input_channels=6)


def test_input_channels_6_constructor_inflates_patch_embed(
    patched_dinomaly_6ch: DinomalyDetector,
) -> None:
    """After construction, the encoder's patch_embed.proj must have in_channels=6
    and be a trainable parameter (the inflation contract)."""
    det = patched_dinomaly_6ch
    proj = det.dinomaly_model.encoder.patch_embed.proj
    assert proj.in_channels == 6
    assert proj.out_channels == 8  # from _FakePatchEmbed
    assert proj.weight.shape == (8, 6, 14, 14)
    # ImageNet stats are tiled to length 6.
    assert len(det.IMAGENET_MEAN) == 6
    assert det.IMAGENET_MEAN[:3] == det.IMAGENET_MEAN[3:]
    # Patch embed is trainable; rest of encoder is frozen.
    assert all(p.requires_grad for p in proj.parameters())
    # The fake encoder has no other parameters, so we just sanity-check the path.
    # In the real DINOv2 encoder, the transformer blocks would still be frozen.


def test_input_channels_6_forward(patched_dinomaly_6ch: DinomalyDetector) -> None:
    """Forward with (B, H, W, 6) input produces (B, H, W, 1) anomaly map."""
    det = patched_dinomaly_6ch
    x = torch.rand(1, 16, 16, 6)
    ctx = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)
    out = det(x, context=ctx)
    assert out["scores"].shape == (1, 16, 16, 1)
    assert out["anomaly_score"].shape == (1,)


def test_input_channels_3_default_still_works(patched_dinomaly: DinomalyDetector) -> None:
    """Backward-compat: omitting input_channels defaults to 3 and the 3-ch path is
    unchanged (no inflation, ImageNet stats length 3)."""
    det = patched_dinomaly
    assert det.input_channels == 3
    assert len(det.IMAGENET_MEAN) == 3
    assert len(det.IMAGENET_STD) == 3


def test_input_channels_invalid_raises() -> None:
    """Non-positive-multiple-of-3 channel counts must raise at construction time."""
    for bad in (0, -3, 4, 5, 7, 8):
        with pytest.raises(ValueError):
            with patch(
                "anomalib.models.image.dinomaly.torch_model.DinomalyModel",
                side_effect=lambda **kw: _FakeDinomalyModelWithPatchEmbed(**kw),
            ):
                DinomalyDetector(encoder_name="fake", input_channels=bad)
