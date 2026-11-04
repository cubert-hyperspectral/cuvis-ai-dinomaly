"""Tests for bounded_topk_mean: exact top-k mean without a TopK above the cap."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_dinomaly.node._bounded_topk_mean import bounded_topk_mean


def _reference(flat: torch.Tensor, k: int) -> torch.Tensor:
    return torch.topk(flat, k, dim=-1).values.mean(dim=-1)


def test_matches_topk_mean_with_k_above_the_cap() -> None:
    torch.manual_seed(0)
    flat = torch.rand(3, 409_600)
    k = 4096  # above MAX_TOPK=3840: needs more than one round
    got = bounded_topk_mean(flat, k, max_topk=3840)
    expected = _reference(flat, k)
    assert torch.allclose(got, expected, atol=1e-6)


def test_matches_topk_mean_with_k_below_the_cap() -> None:
    torch.manual_seed(1)
    flat = torch.rand(2, 65536)
    k = 655  # below MAX_TOPK: a single round
    got = bounded_topk_mean(flat, k, max_topk=3840)
    expected = _reference(flat, k)
    assert torch.allclose(got, expected, atol=1e-6)


def test_matches_topk_mean_with_ties() -> None:
    torch.manual_seed(2)
    base = torch.randint(0, 5, (2, 10_000)).float()  # heavy ties: only 5 distinct values
    k = 4200  # above the cap, with many tied values straddling the cut
    got = bounded_topk_mean(base, k, max_topk=3840)
    expected = _reference(base, k)
    assert torch.allclose(got, expected, atol=1e-6)


def test_exact_three_rounds() -> None:
    torch.manual_seed(3)
    flat = torch.rand(1, 20_000)
    k = 3840 * 2 + 100  # exactly three rounds: 3840, 3840, 100
    got = bounded_topk_mean(flat, k, max_topk=3840)
    expected = _reference(flat, k)
    assert torch.allclose(got, expected, atol=1e-6)


def test_k_equal_to_n_averages_everything() -> None:
    torch.manual_seed(4)
    flat = torch.rand(2, 50)
    got = bounded_topk_mean(flat, 50, max_topk=16)
    assert torch.allclose(got, flat.mean(dim=-1), atol=1e-6)


def test_rejects_k_out_of_range() -> None:
    flat = torch.rand(1, 10)
    with pytest.raises(ValueError):
        bounded_topk_mean(flat, 0)
    with pytest.raises(ValueError):
        bounded_topk_mean(flat, 11)


def test_exported_graph_has_no_topk_above_the_cap_with_multiple_rounds(tmp_path) -> None:
    """A k above max_topk must lower to more than one TopK node, each at or under
    the cap, never a single TopK at the full k."""
    onnx = pytest.importorskip("onnx", reason="onnx extra not installed")
    onnxruntime = pytest.importorskip("onnxruntime", reason="onnx extra not installed")

    class _TopKMean(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return bounded_topk_mean(x, 4096, max_topk=3840)

    module = _TopKMean().eval()
    x = torch.rand(2, 409_600)
    onnx_path = tmp_path / "bounded_topk.onnx"
    torch.onnx.export(
        module,
        (x,),
        str(onnx_path),
        dynamo=True,
        opset_version=18,
        input_names=["x"],
        output_names=["y"],
    )

    model = onnx.load(str(onnx_path))
    by_name = {init.name: init for init in model.graph.initializer}
    topk_values = []
    for node in model.graph.node:
        if node.op_type != "TopK":
            continue
        init = by_name[node.input[1]]
        topk_values.append(int(onnx.numpy_helper.to_array(init).reshape(-1)[0]))

    assert len(topk_values) >= 2, f"expected more than one TopK round, got {topk_values}"
    assert all(k <= 3840 for k in topk_values), topk_values

    with torch.no_grad():
        expected = module(x)
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(["y"], {"x": x.numpy()})[0]
    assert torch.allclose(torch.from_numpy(got), expected, atol=1e-5, rtol=1e-4)


def test_single_round_never_exceeds_max_topk() -> None:
    # Regression guard: a round must never ask torch.topk for more than max_topk,
    # which is the property that keeps the exported ONNX TopK nodes under the cap.
    calls: list[int] = []
    real_topk = torch.topk

    def traced_topk(input_tensor, k, *args, **kwargs):
        calls.append(k)
        return real_topk(input_tensor, k, *args, **kwargs)

    torch.topk = traced_topk
    try:
        flat = torch.rand(1, 100_000)
        bounded_topk_mean(flat, 10_000, max_topk=3840)
    finally:
        torch.topk = real_topk
    assert calls, "torch.topk was never called"
    assert max(calls) <= 3840
