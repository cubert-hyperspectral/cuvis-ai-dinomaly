"""Route anomalib's DINOv2 backbone through the shared model cache and core's registry.

anomalib's ``DinoV2Loader`` hardcodes a CWD-relative ``./pre_trained/`` cache and
downloads the DINOv2 backbone from ``dl.fbaipublicfiles.com``. Two contained,
idempotent patches fix that for Cuvis.AI:

* :func:`redirect_dinov2_cache_to_shared` points the loader's default cache dir
  at ``$CUVIS_MODEL_CACHE_DIR/dinov2`` whenever the orchestrator exports that
  variable (the per-run HOME and CWD are wiped, so the default would re-download
  every run and could never run offline). The variable is read each time a loader
  is constructed, so a process whose environment changes (tests, long-lived
  notebooks) never keeps a stale target.
* :func:`route_dinov2_weights_through_core` replaces the loader's download step:
  the ViT-B/14 reg4 backbone is materialized from cuvis-ai-core's weight registry
  (the ``cubert-gmbh/dinov2`` mirror, commit-pinned and sha256-verified) into the
  loader's cache dir, so nothing is fetched from ``dl.fbaipublicfiles.com`` and an
  offline child without provisioned weights gets core's actionable
  ``ModelWeightsMissingError`` (naming ``download-model download
  dinov2_vitb14_reg4``) instead of a network error. Variants without a mirror keep
  anomalib's original download.

Both only override defaults anomalib gives no hook for (``load_dinov2_model``
constructs ``DinoV2Loader()`` with no arguments); an explicit ``cache_dir`` passed
by a caller is honored.
"""

from __future__ import annotations

import os
from pathlib import Path

# anomalib's hardcoded default; we replace only this sentinel, never an explicit arg.
_DEFAULT_SENTINEL = "./pre_trained/"

# anomalib weight filename -> cuvis-ai-core registry name, for the mirrored variants.
MIRRORED_DINOV2_WEIGHTS: dict[str, str] = {
    "dinov2_vitb14_reg4_pretrain.pth": "dinov2_vitb14_reg4",
}


def _loader_cls() -> type | None:
    try:
        from anomalib.models.image.dinomaly.components import dinov2_loader as _dl
    except Exception:  # anomalib layout changed / not installed: leave defaults
        return None
    return _dl.DinoV2Loader


def shared_dinov2_cache_dir() -> str | None:
    """Return ``$CUVIS_MODEL_CACHE_DIR/dinov2`` for the current environment, or ``None``."""
    root = os.environ.get("CUVIS_MODEL_CACHE_DIR")
    if not root:
        return None
    return str(Path(root) / "dinov2")


def redirect_dinov2_cache_to_shared() -> None:
    """Point ``DinoV2Loader``'s default cache at ``$CUVIS_MODEL_CACHE_DIR/dinov2``.

    Installs the redirect once; every later ``DinoV2Loader()`` construction looks
    the variable up afresh, so an unset variable leaves anomalib's default in place
    and a changed one is honored without re-patching. No-op when anomalib's loader
    is not importable. Idempotent across calls.
    """
    loader_cls = _loader_cls()
    if loader_cls is None or getattr(loader_cls, "_cuvis_cache_patched", False):
        return

    _orig_init = loader_cls.__init__

    def _patched_init(self, cache_dir: str | Path = _DEFAULT_SENTINEL) -> None:
        if str(cache_dir) == _DEFAULT_SENTINEL:
            cache_dir = shared_dinov2_cache_dir() or cache_dir
        _orig_init(self, cache_dir)

    loader_cls.__init__ = _patched_init
    loader_cls._cuvis_cache_patched = True


def route_dinov2_weights_through_core() -> None:
    """Serve the mirrored DINOv2 weights from core's registry instead of downloading.

    Patches ``DinoV2Loader._download_weights`` so the mirrored backbone is
    materialized (hardlink or copy) into the loader's cache dir at exactly the
    point where anomalib would otherwise download it. Unmirrored variants fall
    through to anomalib's download. Idempotent; no-op without anomalib.
    """
    loader_cls = _loader_cls()
    if loader_cls is None or getattr(loader_cls, "_cuvis_weights_routed", False):
        return

    _orig_download = loader_cls._download_weights

    def _patched_download(self, model_type: str, architecture: str, patch_size: int) -> None:
        weight_path = self._get_weight_path(model_type, architecture, patch_size)
        registry_name = MIRRORED_DINOV2_WEIGHTS.get(weight_path.name)
        if registry_name is None:
            _orig_download(self, model_type, architecture, patch_size)
            return
        from cuvis_ai_core.data.model_weights import ModelWeights

        ModelWeights.materialize(registry_name, weight_path.parent, filename=weight_path.name)

    loader_cls._download_weights = _patched_download
    loader_cls._cuvis_weights_routed = True
