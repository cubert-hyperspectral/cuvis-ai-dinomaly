"""Plugin-manifest loading smoke tests for local development manifests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_examples_plugins_manifest_loads_and_registers_nodes() -> None:
    """Ensure NodeRegistry can load the local plugin manifest and resolve classes."""
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    manifest = Path(__file__).resolve().parents[1] / "examples" / "plugins.yaml"
    registry = NodeRegistry()
    registry.register_plugins(str(manifest))

    det = registry.get("DinomalyDetector")
    bridge = registry.get("DinomalyTrainLossBridge")

    assert det.__name__ == "DinomalyDetector"
    assert bridge.__name__ == "DinomalyTrainLossBridge"
