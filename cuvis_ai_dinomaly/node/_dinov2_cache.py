"""Redirect anomalib's DINOv2 backbone cache to the shared model cache.

anomalib's ``DinoV2Loader`` hardcodes a CWD-relative ``./pre_trained/`` cache and
downloads the DINOv2 backbone from ``dl.fbaipublicfiles.com``. Under the cuvis-ai
orchestrator the per-run HOME/CWD is wiped, so the backbone re-downloads every run
and cannot run offline. When the orchestrator exports ``CUVIS_MODEL_CACHE_DIR`` we
monkeypatch the loader's default cache dir to a persistent shared location, so the
backbone is fetched once (or pre-provisioned) and loaded offline thereafter.

This is a contained, idempotent patch of an upstream default we cannot thread a
value through (anomalib's ``load_dinov2_model`` constructs ``DinoV2Loader()`` with
no arguments); it only overrides the hardcoded default and honors an explicit
``cache_dir`` if a caller ever passes one.
"""

from __future__ import annotations

import os
from pathlib import Path

# anomalib's hardcoded default; we replace only this sentinel, never an explicit arg.
_DEFAULT_SENTINEL = "./pre_trained/"


def redirect_dinov2_cache_to_shared() -> None:
    """Point ``DinoV2Loader``'s default cache at ``$CUVIS_MODEL_CACHE_DIR/dinov2``.

    No-op when the env var is unset (local dev keeps anomalib's default) or when
    anomalib's loader is not importable. Idempotent across calls.
    """
    root = os.environ.get("CUVIS_MODEL_CACHE_DIR")
    if not root:
        return
    try:
        from anomalib.models.image.dinomaly.components import dinov2_loader as _dl
    except Exception:  # anomalib layout changed / not installed — leave defaults
        return

    loader_cls = _dl.DinoV2Loader
    # Store/refresh the target so a later env change is honored even after patching.
    loader_cls._cuvis_shared_cache = str(Path(root) / "dinov2")
    if getattr(loader_cls, "_cuvis_cache_patched", False):
        return

    _orig_init = loader_cls.__init__

    def _patched_init(self, cache_dir: str | Path = _DEFAULT_SENTINEL) -> None:
        if str(cache_dir) == _DEFAULT_SENTINEL:
            cache_dir = type(self)._cuvis_shared_cache
        _orig_init(self, cache_dir)

    loader_cls.__init__ = _patched_init
    loader_cls._cuvis_cache_patched = True
