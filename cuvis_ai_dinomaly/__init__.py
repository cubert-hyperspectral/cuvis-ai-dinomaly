"""cuvis-ai-dinomaly: Dinomaly (via Anomalib) plugin for cuvis.ai.

Importing the package registers the plugin's weight declaration (the DINOv2 ViT-B/14
reg4 backbone, :mod:`cuvis_ai_dinomaly.weights`) with cuvis-ai-core's model-weight
registry, so the backbone routing and ``download-model`` in the same environment share
the mirror pin without a plugin manifest on disk.
"""

from cuvis_ai_core.data.model_weights import ModelWeights

from cuvis_ai_dinomaly.node.dinomaly_detector import DinomalyDetector
from cuvis_ai_dinomaly.node.dinomaly_train_loss_bridge import DinomalyTrainLossBridge
from cuvis_ai_dinomaly.weights import PLUGIN_NAME, WEIGHTS

ModelWeights.register(PLUGIN_NAME, WEIGHTS)

__all__ = ["PLUGIN_NAME", "WEIGHTS", "DinomalyDetector", "DinomalyTrainLossBridge"]
