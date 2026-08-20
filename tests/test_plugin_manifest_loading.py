"""Plugin-manifest loading smoke tests for the local development manifest."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_plugin_manifest_loads_and_registers_all_capabilities() -> None:
    """Ensure NodeRegistry loads the manifest and resolves every declared capability."""
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    manifest = Path(__file__).resolve().parents[1] / "configs" / "plugins" / "dinomaly.yaml"
    registry = NodeRegistry()
    registry.register_plugin(str(manifest))

    for class_name in (
        "DinomalyDetector",
        "DinomalyTrainLossBridge",
        "AnomalyAUROCMetrics",
        "PerClassAnomalyAUROC",
        "ValNormalAnomalyMean",
    ):
        assert registry.get(class_name).__name__ == class_name
