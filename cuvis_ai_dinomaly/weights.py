"""Weight declarations of the dinomaly plugin.

Side-effect free on purpose: this module only declares. ``cuvis_ai_dinomaly/__init__``
registers the tuple with cuvis-ai-core's ``ModelWeights`` at import, and cuvis-ai's
``emit_metadata`` projects it into the plugin manifest's ``weights:`` block, so
CuvisNEXT and the installer know what to provision without importing the plugin.

The one row is the DINOv2 ViT-B/14 backbone with four register tokens, which every
``DinomalyDetector`` loads through anomalib's ``DinoV2Loader``
(:mod:`cuvis_ai_dinomaly.node._dinov2_cache` serves it from the shared cache). Other
DINOv2 encoders have no mirror and keep anomalib's download. The Cubert-trained
Dinomaly pipelines are not declared here: cuvis-ai-core owns those rows. The pin comes
from ``tools/mirror_weights.py`` in cuvis-ai-core (the mirror is ``cubert-gmbh/dinov2``,
a byte-identical copy of the file Meta publishes on ``dl.fbaipublicfiles.com``).
"""

from __future__ import annotations

from cuvis_ai_schemas.plugin import PluginWeightEntry

PLUGIN_NAME = "dinomaly"
"""The manifest name of this plugin (what pipelines list under ``plugins:``)."""

WEIGHTS: tuple[PluginWeightEntry, ...] = (
    PluginWeightEntry(
        name="dinov2_vitb14_reg4",
        display_name="DINOv2 ViT-B/14 (reg4)",
        summary="Backbone every Dinomaly detector needs",
        used_for=["Backbone", "Anomaly detection"],
        repo_id="cubert-gmbh/dinov2",
        filename="dinov2_vitb14_reg4_pretrain.pth",
        revision="e2c8060a74112f7537484ed50097b1497d5f032c",
        sha256="73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71",
        size_bytes=346_393_545,
        license="Apache-2.0",
        license_file="LICENSE",
        description=(
            "DINOv2 ViT-B/14 backbone with four register tokens; every DinomalyDetector "
            "loads it through anomalib's DinoV2Loader, and training a Dinomaly pipeline "
            "starts from it (mirror of Meta's dinov2_vitb14_reg4_pretrain.pth, unmodified)."
        ),
    ),
)
"""Every weight the dinomaly nodes load from the shared cache."""
