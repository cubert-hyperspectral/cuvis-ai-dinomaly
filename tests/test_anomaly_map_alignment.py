"""Anomaly-map alignment: the internal patch upsample must be area-based.

anomalib's DinomalyModel upsamples the patch anomaly map with align_corners=True, which
shifts the returned map radially outward relative to the (area-based) preprocessing and the
final scores->native resize. ``_area_anomaly_map_upsample`` corrects that by forcing
align_corners=False on anomalib's internal ``F.interpolate`` for the duration of the model
call. These tests exercise the context manager directly (no model download).
"""

from __future__ import annotations

import torch
from test_fast_inference import _make_detector

from cuvis_ai_dinomaly.node.dinomaly_detector import _AREA_ALIGN_F, _area_anomaly_map_upsample


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
        # a request already asking for False (or omitting align_corners) is untouched
        got_false = tm.F.interpolate(x, size=(9, 9), mode="bilinear", align_corners=False)
    assert torch.allclose(got, up_false)  # a True request was served as False inside
    assert torch.allclose(got_false, up_false)

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


def test_installed_proxy_is_identity_stable():
    # torch.compile guards on the identity of the globals it traced through, so every
    # enable must install the SAME proxy object, and nesting must not stack proxies.
    tm = _tm()
    real_f = tm.F
    with _area_anomaly_map_upsample(True):
        assert tm.F is _AREA_ALIGN_F
        with _area_anomaly_map_upsample(True):
            assert tm.F is _AREA_ALIGN_F  # re-entrant: no proxy-on-proxy
    with _area_anomaly_map_upsample(True):
        assert tm.F is _AREA_ALIGN_F  # same object across separate enables
    assert tm.F is real_f


def test_align_map_to_input_serialized_in_hparams():
    # The opt-out must survive a pipeline save/restore: hparams record what was passed,
    # not the signature default.
    det = _make_detector(image_size=448, crop_size=392, align_map_to_input=False)
    assert det.align_map_to_input is False
    assert det.hparams["align_map_to_input"] is False

    det_default = _make_detector(image_size=448, crop_size=392)
    assert det_default.align_map_to_input is True
    assert det_default.hparams["align_map_to_input"] is True
