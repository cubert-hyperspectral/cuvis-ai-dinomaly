"""Tests for the patch-embed weight inflation utility.

The activation-parity test is the meaningful one: it proves the ``/factor`` math is
correct end-to-end on a real ``Conv2d`` forward, so the inflated DINOv2 patch embed
will produce identical pre-bottleneck features at init when fed a duplicated input.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cuvis_ai_dinomaly.node._patch_embed_inflation import inflate_conv2d_input_channels


@pytest.fixture
def base_proj() -> nn.Conv2d:
    """A patch-embed shaped like DINOv2 ViT-B/14: 3 in, 768 out, kernel/stride 14, no bias."""
    torch.manual_seed(0)
    return nn.Conv2d(in_channels=3, out_channels=768, kernel_size=14, stride=14, bias=False)


@pytest.fixture
def base_proj_with_bias() -> nn.Conv2d:
    torch.manual_seed(0)
    return nn.Conv2d(in_channels=3, out_channels=32, kernel_size=14, stride=14, bias=True)


def test_inflated_shape_and_metadata(base_proj: nn.Conv2d) -> None:
    new = inflate_conv2d_input_channels(base_proj, new_in_channels=6)
    assert new.in_channels == 6
    assert new.out_channels == base_proj.out_channels
    assert new.kernel_size == base_proj.kernel_size
    assert new.stride == base_proj.stride
    assert new.weight.shape == (768, 6, 14, 14)
    # Original is untouched.
    assert base_proj.weight.shape == (768, 3, 14, 14)


def test_activation_parity_on_duplicated_input(base_proj: nn.Conv2d) -> None:
    """Feeding ``x.repeat(1, 2, 1, 1)`` through the inflated conv must match
    feeding ``x`` through the original conv. Proves the /factor math."""
    new = inflate_conv2d_input_channels(base_proj, new_in_channels=6)
    x3 = torch.randn(2, 3, 28, 28)  # 28 = 2 patches at stride 14
    x6 = x3.repeat(1, 2, 1, 1)
    y_old = base_proj(x3)
    y_new = new(x6)
    assert y_old.shape == y_new.shape
    assert torch.allclose(y_old, y_new, atol=1e-5, rtol=1e-5)


def test_bias_is_copied_verbatim(base_proj_with_bias: nn.Conv2d) -> None:
    new = inflate_conv2d_input_channels(base_proj_with_bias, new_in_channels=6)
    assert new.bias is not None
    assert torch.equal(new.bias, base_proj_with_bias.bias)


def test_bias_absent_when_old_had_no_bias(base_proj: nn.Conv2d) -> None:
    new = inflate_conv2d_input_channels(base_proj, new_in_channels=6)
    assert new.bias is None


def test_factor_three_inflation_to_9_channels(base_proj: nn.Conv2d) -> None:
    """Sanity that any positive integer factor works, not just 2."""
    new = inflate_conv2d_input_channels(base_proj, new_in_channels=9)
    assert new.weight.shape == (768, 9, 14, 14)
    x3 = torch.randn(1, 3, 28, 28)
    x9 = x3.repeat(1, 3, 1, 1)
    assert torch.allclose(base_proj(x3), new(x9), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("new_in", [0, -1, 4, 5, 7])  # 4,5,7 are not multiples of 3
def test_rejects_invalid_channel_counts(base_proj: nn.Conv2d, new_in: int) -> None:
    with pytest.raises(ValueError):
        inflate_conv2d_input_channels(base_proj, new_in_channels=new_in)


def test_identity_inflation_factor_one(base_proj: nn.Conv2d) -> None:
    """factor=1 (new_in_channels == old_in_channels) is a degenerate but legal case
    that returns a copy with identical weights (divided by 1 = unchanged)."""
    new = inflate_conv2d_input_channels(base_proj, new_in_channels=3)
    assert torch.equal(new.weight, base_proj.weight)
