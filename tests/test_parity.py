"""Numerical parity vs raw ``DinomalyModel`` (optional slow / network)."""

import os

import pytest
import torch
import torch.nn.functional as F
from anomalib.models.image.dinomaly.torch_model import DinomalyModel
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

from cuvis_ai_dinomaly.node.dinomaly_detector import DinomalyDetector

requires_weights = pytest.mark.skipif(
    os.environ.get("CUVIS_DINOMALY_SKIP_SLOW", "") == "1",
    reason="CUVIS_DINOMALY_SKIP_SLOW=1",
)


@requires_weights
@pytest.mark.slow
def test_training_loss_matches_raw_model() -> None:
    torch.manual_seed(0)
    enc = "dinov2reg_vit_small_14"
    raw = DinomalyModel(encoder_name=enc)
    det = DinomalyDetector(encoder_name=enc)
    det.dinomaly_model.load_state_dict(raw.state_dict())

    x = torch.rand(1, 128, 128, 3)
    ctx = Context(stage=ExecutionStage.TRAIN, epoch=0, batch_idx=0, global_step=42)
    x_chw = det._rgb_bhwc_to_model_input(x)
    raw.train()
    ref_loss = raw(x_chw, global_step=42)
    out = det(x, context=ctx)
    # Dropout in the bottleneck differs across module instances / step order; keep moderate tolerance.
    assert torch.allclose(out["training_loss"], ref_loss, rtol=0.05, atol=0.05)


@requires_weights
@pytest.mark.slow
def test_inference_matches_raw_model() -> None:
    torch.manual_seed(1)
    enc = "dinov2reg_vit_small_14"
    raw = DinomalyModel(encoder_name=enc)
    det = DinomalyDetector(encoder_name=enc)
    det.dinomaly_model.load_state_dict(raw.state_dict())

    x = torch.rand(1, 128, 128, 3)
    x_chw = det._rgb_bhwc_to_model_input(x)
    raw.eval()
    with torch.no_grad():
        pred = raw(x_chw)
    ctx = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)
    out = det(x, context=ctx)

    assert torch.allclose(out["anomaly_score"], pred.pred_score, rtol=1e-4, atol=1e-4)
    am = pred.anomaly_map
    if am.dim() == 3:
        am = am.unsqueeze(1)
    h, w = x.shape[1], x.shape[2]
    am_hw = F.interpolate(am, size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
    assert torch.allclose(out["scores"].squeeze(-1), am_hw, rtol=1e-3, atol=1e-3)
