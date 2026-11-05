"""Tests for DinomalyDetector.export_module(): the ONNX span-export adapter.

Covers:
- the adapter reproduces the node's own inference forward on seeded input
- the antialiased-resize operator matches torchvision within float32 rounding
- the adapter exports with the dynamo exporter and matches ONNX Runtime
- the exported graph carries no antialiased Resize node
- (slow) the same two checks against the real DINOv2 backbone, when cached
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

from cuvis_ai_dinomaly.node._antialias_resize_matrix import AntialiasResizeMatrix
from cuvis_ai_dinomaly.node.dinomaly_detector import DinomalyDetector

onnxscript = pytest.importorskip("onnxscript", reason="onnx extra not installed")
onnxruntime = pytest.importorskip("onnxruntime", reason="onnx extra not installed")
onnx = pytest.importorskip("onnx", reason="onnx extra not installed")

from torchvision.transforms import InterpolationMode  # noqa: E402
from torchvision.transforms.v2.functional import resize  # noqa: E402


class _FakePatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 8, kernel_size=14, stride=14, bias=False)
        self.in_chans = 3


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _FakePatchEmbed()


class _FakeDinomalyModel(nn.Module):
    """Deterministic stand-in with the real model's score-path shape: a patch-level
    anomaly map keyed off the mean input value plus a ramp (many distinct, strictly
    ordered values, so a top-k check is not degenerate), ``get_encoder_decoder_outputs``
    / ``calculate_anomaly_maps`` / ``gaussian_blur`` exposed the way
    ``_DetectorExportAdapter`` calls them, and a ``forward`` that reproduces anomalib's
    own inference branch (resize to 256, blur, sorted-mean score) from those same
    primitives, so the eager node, the export adapter and ONNX Runtime must all compute
    the exact same numbers (no real DINOv2 weights needed)."""

    def __init__(self, **_: object) -> None:
        super().__init__()
        self.encoder = _FakeEncoder()
        self.bottleneck = nn.Linear(1, 1, bias=False)
        self.decoder = nn.Linear(1, 1, bias=False)
        self.gaussian_blur = nn.Identity()

    def get_encoder_decoder_outputs(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        h, w = x.shape[2], x.shape[3]
        hp, wp = h // 14, w // 14
        base = x.mean(dim=(1, 2, 3))  # [B]
        feat = base.reshape(x.shape[0], 1, 1).expand(x.shape[0], hp, wp)
        return [feat], [feat]

    @staticmethod
    def calculate_anomaly_maps(
        source_feature_maps: list[torch.Tensor],
        target_feature_maps: list[torch.Tensor],
        out_size: int | tuple[int, int] = 392,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if not isinstance(out_size, tuple):
            out_size = (int(out_size), int(out_size))
        feat = source_feature_maps[0]
        hp, wp = feat.shape[-2], feat.shape[-1]
        ramp = torch.linspace(0.0, 1.0, hp * wp, device=feat.device, dtype=feat.dtype).reshape(
            hp, wp
        )
        amap = (feat + ramp.unsqueeze(0)).unsqueeze(1)  # [B, 1, hp, wp]
        amap = torch.nn.functional.interpolate(
            amap, size=out_size, mode="bilinear", align_corners=True
        )
        return amap, [amap]

    def forward(
        self, x: torch.Tensor, global_step: int | None = None
    ) -> torch.Tensor | SimpleNamespace:
        if global_step is not None:
            return x.mean().reshape(())
        en, de = self.get_encoder_decoder_outputs(x)
        amap, _ = self.calculate_anomaly_maps(en, de, out_size=x.shape[-2:])
        resized = torch.nn.functional.interpolate(
            amap, size=256, mode="bilinear", align_corners=False
        )
        blurred = self.gaussian_blur(resized)
        flat = blurred.flatten(1)
        k = int(flat.shape[1] * 0.01)
        score = torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(dim=1)
        return SimpleNamespace(anomaly_map=amap, pred_score=score)


def _make_detector(**kw) -> DinomalyDetector:
    from unittest.mock import patch

    with patch(
        "anomalib.models.image.dinomaly.torch_model.DinomalyModel",
        side_effect=lambda **k: _FakeDinomalyModel(**k),
    ):
        return DinomalyDetector(encoder_name="fake", **kw)


# ---------------------------------------------------------------------------
# (a) adapter equals node forward
# ---------------------------------------------------------------------------


def test_export_module_matches_node_forward_on_seeded_input() -> None:
    torch.manual_seed(0)
    det = _make_detector(image_size=56, crop_size=56, use_center_crop=False)
    det.eval()
    x = torch.rand(2, 56, 56, 3)

    ctx = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)
    expected = det(x, context=ctx)

    adapter = det.export_module()
    assert adapter is not None
    adapter.eval()
    with torch.no_grad():
        got = adapter(rgb_image=x)

    assert torch.allclose(got["scores"], expected["scores"], atol=1e-6)
    assert torch.allclose(got["anomaly_score"], expected["anomaly_score"], atol=1e-6)


def test_export_module_matches_node_forward_on_uint8_input() -> None:
    torch.manual_seed(1)
    det = _make_detector(image_size=56, crop_size=56, use_center_crop=False)
    det.eval()
    x = torch.randint(0, 256, (1, 56, 56, 3), dtype=torch.uint8)

    ctx = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)
    expected = det(x, context=ctx)

    adapter = det.export_module()
    adapter.eval()
    with torch.no_grad():
        got = adapter(rgb_image=x.float())

    assert torch.allclose(got["scores"], expected["scores"], atol=1e-6)
    assert torch.allclose(got["anomaly_score"], expected["anomaly_score"], atol=1e-6)


def test_export_module_tag_declared() -> None:
    from cuvis_ai_schemas.enums import NodeTag

    assert NodeTag.ONNX in DinomalyDetector.get_tags()


# ---------------------------------------------------------------------------
# (b) the resize operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("h_in", "w_in", "h_out", "w_out"),
    [
        (448, 448, 392, 392),  # downscale
        (100, 120, 448, 448),  # upscale
        (600, 500, 448, 448),  # downscale, non-square
    ],
)
def test_antialias_resize_matrix_matches_torchvision(h_in, w_in, h_out, w_out) -> None:
    torch.manual_seed(0)
    x = torch.rand(2, 3, h_in, w_in)
    expected = resize(x, [h_out, w_out], interpolation=InterpolationMode.BILINEAR, antialias=True)

    op = AntialiasResizeMatrix((h_out, w_out))
    got = op(x)

    # torchvision's own float32 antialiased kernel differs from the float64
    # reference our matrices are built against by ~3e-5 at a typical downscale
    # (verified against a float64 torchvision run); not an error in the matrix
    # construction, which matches that reference to ~6e-8.
    assert torch.allclose(got, expected, atol=5e-5, rtol=0)


def test_antialias_resize_matrix_rejects_untraced_unbuilt_call() -> None:
    op = AntialiasResizeMatrix((392, 392))
    with torch.compiler.set_stance("force_eager"):
        pass  # sanity: stance API exists in this torch; the real guard uses is_compiling()

    # Simulate "called under tracing before being built" directly: this is the
    # condition export_module() must avoid hitting by running an eager forward
    # first. Patch the compiling check instead of actually tracing, to keep
    # this test independent of the dynamo exporter.
    import unittest.mock as mock

    with mock.patch("torch.compiler.is_compiling", return_value=True):
        with pytest.raises(RuntimeError, match="not built yet"):
            op(torch.rand(1, 3, 56, 56))


def test_antialias_resize_matrix_rejects_shape_change_after_build() -> None:
    op = AntialiasResizeMatrix((56, 56))
    op(torch.rand(1, 3, 56, 56))
    with pytest.raises(RuntimeError, match="fixed to one input size"):
        op(torch.rand(1, 3, 64, 64))


# ---------------------------------------------------------------------------
# (c) + (d) dynamo export vs ONNX Runtime, no antialiased Resize node
# ---------------------------------------------------------------------------


def test_export_module_exports_and_matches_onnxruntime(tmp_path) -> None:
    torch.manual_seed(2)
    det = _make_detector(image_size=56, crop_size=56, use_center_crop=False)
    det.eval()
    x = torch.rand(1, 56, 56, 3)

    adapter = det.export_module()
    adapter.eval()
    with torch.no_grad():
        adapter(rgb_image=x)  # eager warm-up call: builds the resize matrices

    onnx_path = tmp_path / "detector_adapter.onnx"
    torch.onnx.export(
        adapter,
        (),
        str(onnx_path),
        kwargs={"rgb_image": x},
        dynamo=True,
        opset_version=18,
        input_names=["rgb_image"],
        output_names=["scores", "anomaly_score"],
    )

    with torch.no_grad():
        expected = adapter(rgb_image=x)

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got_scores, got_anomaly_score = session.run(
        ["scores", "anomaly_score"], {"rgb_image": x.numpy()}
    )

    assert torch.allclose(torch.from_numpy(got_scores), expected["scores"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(
        torch.from_numpy(got_anomaly_score), expected["anomaly_score"], atol=1e-5, rtol=1e-4
    )

    model = onnx.load(str(onnx_path))
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        for attr in node.attribute:
            if attr.name == "antialias":
                assert attr.i == 0, "the exported graph must not antialias its Resize node"


def _topk_initializer_values(model: onnx.ModelProto) -> list[int]:
    """The ``K`` value of every ``TopK`` node in ``model``, read off its initializer."""
    by_name = {init.name: init for init in model.graph.initializer}
    values: list[int] = []
    for node in model.graph.node:
        if node.op_type != "TopK":
            continue
        k_input = node.input[1]
        init = by_name.get(k_input)
        assert init is not None, f"TopK node {node.name!r} has no initializer for K"
        arr = onnx.numpy_helper.to_array(init)
        values.append(int(arr.reshape(-1)[0]))
    return values


def test_exported_graph_has_no_topk_node_above_the_tensorrt_cap(tmp_path) -> None:
    """Regression test for the TensorRT ITopKLayer K=3840 cap: every TopK node the
    exported graph carries (the image-score top-k, in however many rounds it took)
    must ask for at most 3840 values, never the full sort anomalib's own code would
    lower to."""
    from cuvis_ai_dinomaly.node._bounded_topk_mean import MAX_TOPK

    torch.manual_seed(5)
    det = _make_detector(image_size=56, crop_size=56, use_center_crop=False)
    det.eval()
    x = torch.rand(1, 56, 56, 3)

    adapter = det.export_module()
    adapter.eval()
    with torch.no_grad():
        adapter(rgb_image=x)

    onnx_path = tmp_path / "detector_adapter_topk.onnx"
    torch.onnx.export(
        adapter,
        (),
        str(onnx_path),
        kwargs={"rgb_image": x},
        dynamo=True,
        opset_version=18,
        input_names=["rgb_image"],
        output_names=["scores", "anomaly_score"],
    )

    model = onnx.load(str(onnx_path))
    topk_values = _topk_initializer_values(model)
    assert topk_values, "expected at least one TopK node for the image score"
    assert all(k <= MAX_TOPK for k in topk_values), (
        f"TopK node(s) above the TensorRT cap of {MAX_TOPK}: {topk_values}"
    )


# ---------------------------------------------------------------------------
# (e) slow: the real DINOv2 backbone, when cached
# ---------------------------------------------------------------------------

requires_weights = pytest.mark.skipif(
    os.environ.get("CUVIS_DINOMALY_SKIP_SLOW", "") == "1",
    reason="CUVIS_DINOMALY_SKIP_SLOW=1",
)


def _dinov2_cached() -> bool:
    from cuvis_ai_core.data.model_weights import ModelWeights

    try:
        status = ModelWeights.status("dinov2_vitb14_reg4")
    except Exception:
        return False
    return bool(getattr(status, "present", False))


@requires_weights
@pytest.mark.slow
@pytest.mark.skipif(not _dinov2_cached(), reason="dinov2_vitb14_reg4 not in the local HF cache")
def test_export_module_matches_real_backbone(tmp_path) -> None:
    torch.manual_seed(3)
    det = DinomalyDetector(
        encoder_name="dinov2reg_vit_base_14",
        image_size=56,
        crop_size=56,
        use_center_crop=False,
    )
    det.eval()
    x = torch.rand(1, 56, 56, 3)

    ctx = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)
    with torch.no_grad():
        expected = det(x, context=ctx)

    adapter = det.export_module()
    adapter.eval()
    with torch.no_grad():
        got = adapter(rgb_image=x)
    assert torch.allclose(got["scores"], expected["scores"], atol=1e-6)
    assert torch.allclose(got["anomaly_score"], expected["anomaly_score"], atol=1e-6)

    onnx_path = tmp_path / "real_detector_adapter.onnx"
    torch.onnx.export(
        adapter,
        (),
        str(onnx_path),
        kwargs={"rgb_image": x},
        dynamo=True,
        opset_version=18,
        input_names=["rgb_image"],
        output_names=["scores", "anomaly_score"],
    )
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got_scores, got_anomaly_score = session.run(
        ["scores", "anomaly_score"], {"rgb_image": x.numpy()}
    )
    assert torch.allclose(torch.from_numpy(got_scores), got["scores"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(
        torch.from_numpy(got_anomaly_score), got["anomaly_score"], atol=1e-5, rtol=1e-4
    )
