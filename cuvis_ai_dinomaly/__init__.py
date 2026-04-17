"""cuvis-ai-dinomaly: Dinomaly (via Anomalib) plugin for cuvis.ai."""

from cuvis_ai_dinomaly.node.dinomaly_detector import DinomalyDetector
from cuvis_ai_dinomaly.node.dinomaly_train_loss_bridge import DinomalyTrainLossBridge

__all__ = ["DinomalyDetector", "DinomalyTrainLossBridge"]
