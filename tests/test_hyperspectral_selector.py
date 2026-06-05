"""Tests for FixedHyperspectralSelector — verifies arbitrary-length wavelength picks
and that the output channel order matches the target wavelength order."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai_dinomaly.node.selectors import FixedHyperspectralSelector


@pytest.fixture
def bedding_cube() -> tuple[torch.Tensor, np.ndarray]:
    """Construct a synthetic bedding-shaped cube where each band's pixel value is
    a fixed multiple of (band_index + 1). Lets us verify the selector picked the
    right indices just by inspecting the output values."""
    wavelengths = np.array([450, 550, 625, 1050, 1200, 1450], dtype=np.int32)
    # Each band gets a constant value (band_index + 1) so we can trace identity.
    H, W = 8, 8
    cube = torch.zeros(1, H, W, 6, dtype=torch.float32)
    for i in range(6):
        cube[..., i] = float(i + 1)
    return cube, wavelengths


def test_6_channel_target_descending_pairs_order(bedding_cube) -> None:
    """bedding-all6 layout: (625, 550, 450, 1450, 1200, 1050) — descending λ within each
    triplet. Output channel order must match the target order exactly."""
    cube, wavelengths = bedding_cube
    sel = FixedHyperspectralSelector(target_wavelengths=(625, 550, 450, 1450, 1200, 1050))
    out = sel.forward(cube=cube, wavelengths=wavelengths)
    img = out["rgb_image"]
    info = out["band_info"]
    assert img.shape == (1, 8, 8, 6)
    # bands index_in_cube = [2, 1, 0, 5, 4, 3], values = index+1 = [3, 2, 1, 6, 5, 4]
    expected_values = [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]
    for ch, ev in enumerate(expected_values):
        assert torch.all(img[..., ch] == ev), f"ch={ch}: expected {ev} got {img[..., ch].unique()}"
    assert info["band_indices"] == [2, 1, 0, 5, 4, 3]
    assert info["target_wavelengths_nm"] == [625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0]
    assert info["normalized_output"] is False


def test_3_channel_target_backward_compat_with_rgb(bedding_cube) -> None:
    """Default 3-channel RGB-equivalent target works and matches FixedWavelengthSelector
    semantics: bands stacked in target order."""
    cube, wavelengths = bedding_cube
    sel = FixedHyperspectralSelector(target_wavelengths=(625.0, 550.0, 450.0))
    out = sel.forward(cube=cube, wavelengths=wavelengths)
    img = out["rgb_image"]
    assert img.shape == (1, 8, 8, 3)
    assert torch.all(img[..., 0] == 3.0)   # 625 nm → band 2
    assert torch.all(img[..., 1] == 2.0)   # 550 nm → band 1
    assert torch.all(img[..., 2] == 1.0)   # 450 nm → band 0


def test_nearest_match_for_off_grid_wavelengths(bedding_cube) -> None:
    """Targets that don't exactly match an input wavelength snap to the nearest band."""
    cube, wavelengths = bedding_cube
    sel = FixedHyperspectralSelector(target_wavelengths=(640.0, 1490.0))  # → 625 (band 2), 1450 (band 5)
    out = sel.forward(cube=cube, wavelengths=wavelengths)
    img = out["rgb_image"]
    assert img.shape == (1, 8, 8, 2)
    assert torch.all(img[..., 0] == 3.0)
    assert torch.all(img[..., 1] == 6.0)


def test_normalize_output_divides_by_max(bedding_cube) -> None:
    """With normalize_output=True, the global max becomes 1.0."""
    cube, wavelengths = bedding_cube
    sel = FixedHyperspectralSelector(
        target_wavelengths=(625.0, 1450.0), normalize_output=True
    )
    out = sel.forward(cube=cube, wavelengths=wavelengths)
    img = out["rgb_image"]
    # max value pre-norm is 6 (the 1450 nm band), so after /6 the 1450 channel is 1.0
    # and the 625 channel (value 3) is 0.5.
    assert float(img.amax()) == pytest.approx(1.0)
    assert torch.allclose(img[..., 0], torch.full_like(img[..., 0], 0.5))
    assert torch.allclose(img[..., 1], torch.full_like(img[..., 1], 1.0))


def test_single_wavelength_target(bedding_cube) -> None:
    """A single-band selector emits a (B, H, W, 1) output — useful for sanity smoke tests."""
    cube, wavelengths = bedding_cube
    sel = FixedHyperspectralSelector(target_wavelengths=(1050.0,))
    out = sel.forward(cube=cube, wavelengths=wavelengths)
    assert out["rgb_image"].shape == (1, 8, 8, 1)
    assert torch.all(out["rgb_image"][..., 0] == 4.0)  # band 3


def test_empty_target_wavelengths_raises() -> None:
    with pytest.raises(ValueError):
        FixedHyperspectralSelector(target_wavelengths=())
