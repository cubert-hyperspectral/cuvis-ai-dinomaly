"""Tests for the DINOv2 backbone cache redirect (network-free).

Instantiating ``DinoV2Loader`` only mkdirs the cache dir; it does not download
(that happens in ``.load()``), so these assert the redirect without network.
"""

from __future__ import annotations

from cuvis_ai_dinomaly.node._dinov2_cache import redirect_dinov2_cache_to_shared


def test_noop_when_env_unset(monkeypatch):
    monkeypatch.delenv("CUVIS_MODEL_CACHE_DIR", raising=False)
    redirect_dinov2_cache_to_shared()  # must not raise


def test_redirects_loader_default_to_shared(monkeypatch, tmp_path):
    from anomalib.models.image.dinomaly.components import dinov2_loader as dl

    monkeypatch.setenv("CUVIS_MODEL_CACHE_DIR", str(tmp_path / "mc"))
    redirect_dinov2_cache_to_shared()
    loader = dl.DinoV2Loader()
    assert loader.cache_dir == tmp_path / "mc" / "dinov2"


def test_honors_explicit_cache_dir(monkeypatch, tmp_path):
    from anomalib.models.image.dinomaly.components import dinov2_loader as dl

    monkeypatch.setenv("CUVIS_MODEL_CACHE_DIR", str(tmp_path / "mc"))
    redirect_dinov2_cache_to_shared()
    loader = dl.DinoV2Loader(cache_dir=str(tmp_path / "explicit"))
    assert loader.cache_dir == tmp_path / "explicit"
