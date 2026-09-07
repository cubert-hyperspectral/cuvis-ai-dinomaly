"""Back-compat shim: ``AnomalyAUROCMetrics`` now lives upstream in cuvis-ai.

The streaming pixel/image AUROC node was moved to ``cuvis_ai.node.metrics.AnomalyAUROCMetrics``
so any pipeline can wire it without depending on this plugin (ALL-5851). This module re-exports
the upstream class under its historical import path,
``cuvis_ai_dinomaly.node.auroc_metrics.AnomalyAUROCMetrics``, so pipeline YAMLs saved against the
old path keep loading. New pipelines should reference the upstream path directly.

Requires cuvis-ai >= 0.16.0 (the release that ships the node upstream).
"""

from __future__ import annotations

from cuvis_ai.node.metrics import AnomalyAUROCMetrics

__all__ = ["AnomalyAUROCMetrics"]
