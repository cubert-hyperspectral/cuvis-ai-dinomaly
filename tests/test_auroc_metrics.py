"""The AUROC node moved to cuvis-ai; this asserts the back-compat shim still resolves.

``AnomalyAUROCMetrics`` now lives at ``cuvis_ai.node.metrics.AnomalyAUROCMetrics`` (ALL-5851);
its behaviour is tested there. Here we only guard the historical import path
``cuvis_ai_dinomaly.node.auroc_metrics.AnomalyAUROCMetrics`` — the one saved pipeline YAMLs
reference — so it keeps loading. The shim imports the high-level ``cuvis-ai`` package, which is
declared only in the ``examples`` extra, so this is skipped where cuvis-ai is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cuvis_ai.node.metrics", reason="cuvis-ai (examples extra) not installed")


def test_old_import_path_reexports_upstream_class() -> None:
    """The plugin path resolves to the exact upstream class object."""
    from cuvis_ai.node.metrics import AnomalyAUROCMetrics as Upstream
    from cuvis_ai_dinomaly.node.auroc_metrics import AnomalyAUROCMetrics as Shim

    assert Shim is Upstream
