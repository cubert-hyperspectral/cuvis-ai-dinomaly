"""Anomaly-map alignment: the internal patch upsample must be area-based.

anomalib's DinomalyModel upsamples the patch anomaly map with align_corners=True, which
shifts the returned map radially outward relative to the (area-based) preprocessing and the
final scores->native resize. ``_area_anomaly_map_upsample`` corrects that by forcing
align_corners=False on anomalib's internal ``F.interpolate`` for the duration of the model
call. These tests exercise the context manager directly (no model download).
"""

from __future__ import annotations

import torch

from cuvis_ai_dinomaly.node.dinomaly_detector import _area_anomaly_map_upsample


def _tm():
    import anomalib.models.image.dinomaly.torch_model as tm

    return tm


def test_enabled_forces_align_corners_false_and_restores():
    tm = _tm()
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    up_true = tm.F.interpolate(x, size=(9, 9), mode="bilinear", align_corners=True)
    up_false = tm.F.interpolate(x, size=(9, 9), mode="bilinear", align_corners=False)
    assert not torch.allclose(up_true, up_false)  # the two conventions genuinely differ

    with _area_anomaly_map_upsample(True):
        got = tm.F.interpolate(x, size=(9, 9), mode="bilinear", align_corners=True)
    assert torch.allclose(got, up_false)  # a True request was served as False inside

    # module F is restored on exit
    after = tm.F.interpolate(x, size=(9, 9), mode="bilinear", align_corners=True)
    assert torch.allclose(after, up_true)


def test_disabled_is_passthrough():
    tm = _tm()
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    up_true = tm.F.interpolate(x, size=(9, 9), mode="bilinear", align_corners=True)
    with _area_anomaly_map_upsample(False):
        got = tm.F.interpolate(x, size=(9, 9), mode="bilinear", align_corners=True)
    assert torch.allclose(got, up_true)


def test_non_interpolate_attributes_delegate():
    tm = _tm()
    x = torch.randn(2, 3)
    with _area_anomaly_map_upsample(True):
        # a request that does not pass align_corners is untouched, and other F.* still work
        assert torch.allclose(tm.F.relu(x), torch.relu(x))
