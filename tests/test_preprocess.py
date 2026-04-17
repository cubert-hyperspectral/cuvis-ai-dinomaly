"""Preprocessing matches Anomalib Dinomaly defaults (Resize, CenterCrop, Normalize)."""

import torch
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from cuvis_ai_dinomaly.node.dinomaly_detector import DinomalyDetector


def test_preprocess_matches_reference_compose() -> None:
    image_size, crop_size = 448, 392
    ref = Compose(
        [
            Resize((image_size, image_size)),
            CenterCrop(crop_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    det = DinomalyDetector(image_size=image_size, crop_size=crop_size)
    bhwc = torch.rand(2, 100, 120, 3)
    x01 = bhwc.clamp(0, 1)
    bchw = x01.permute(0, 3, 1, 2)
    expected = ref(bchw)
    got = det._rgb_bhwc_to_model_input(x01)
    assert torch.allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_preprocess_without_center_crop_matches_reference() -> None:
    image_size = 448
    ref = Compose(
        [
            Resize((image_size, image_size)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    det = DinomalyDetector(image_size=image_size, crop_size=392, use_center_crop=False)
    bhwc = torch.rand(2, 100, 120, 3)
    x01 = bhwc.clamp(0, 1)
    bchw = x01.permute(0, 3, 1, 2)
    expected = ref(bchw)
    got = det._rgb_bhwc_to_model_input(x01)
    assert torch.allclose(got, expected, rtol=1e-5, atol=1e-5)
