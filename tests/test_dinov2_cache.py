"""Tests for the DINOv2 backbone cache redirect and the core weight routing (network-free).

Instantiating ``DinoV2Loader`` only mkdirs the cache dir; downloading happens in
``_download_weights``, which is patched to go through cuvis-ai-core's registry.
``ModelWeights.materialize`` and anomalib's ``urlretrieve`` are stubbed here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from cuvis_ai_core.data.model_weights import ModelWeights, ModelWeightsMissingError

from cuvis_ai_dinomaly.node._dinov2_cache import (
    redirect_dinov2_cache_to_shared,
    route_dinov2_weights_through_core,
)


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


def _never_download(monkeypatch, dl) -> None:
    def _fail(*args, **kwargs):
        pytest.fail("anomalib must not download from dl.fbaipublicfiles.com")

    monkeypatch.setattr(dl, "urlretrieve", _fail)


def test_mirrored_variant_is_materialized_not_downloaded(monkeypatch, tmp_path):
    from anomalib.models.image.dinomaly.components import dinov2_loader as dl

    route_dinov2_weights_through_core()
    _never_download(monkeypatch, dl)
    calls: list[tuple[str, Path, str | None]] = []

    def fake_materialize(cls, name, dest_dir, *, filename=None, **kwargs):
        calls.append((name, Path(dest_dir), filename))
        target = Path(dest_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"dino")
        return target

    monkeypatch.setattr(ModelWeights, "materialize", classmethod(fake_materialize))

    loader = dl.DinoV2Loader(cache_dir=str(tmp_path / "cache"))
    loader._download_weights("dinov2_reg", "base", 14)

    assert calls == [("dinov2_vitb14_reg4", tmp_path / "cache", "dinov2_vitb14_reg4_pretrain.pth")]
    assert (tmp_path / "cache" / "dinov2_vitb14_reg4_pretrain.pth").read_bytes() == b"dino"


def test_unmirrored_variant_keeps_anomalib_download(monkeypatch, tmp_path):
    from anomalib.models.image.dinomaly.components import dinov2_loader as dl

    route_dinov2_weights_through_core()
    fetched: list[str] = []

    def fake_urlretrieve(url, filename, reporthook=None):
        fetched.append(url)
        Path(filename).write_bytes(b"small")

    monkeypatch.setattr(dl, "urlretrieve", fake_urlretrieve)

    def _fail(cls, *args, **kwargs):
        pytest.fail("materialize must not be called for an unmirrored variant")

    monkeypatch.setattr(ModelWeights, "materialize", classmethod(_fail))

    loader = dl.DinoV2Loader(cache_dir=str(tmp_path / "cache"))
    loader._download_weights("dinov2_reg", "small", 14)

    assert len(fetched) == 1
    assert fetched[0].endswith("dinov2_vits14/dinov2_vits14_reg4_pretrain.pth")


def test_missing_weights_error_reaches_the_caller(monkeypatch, tmp_path):
    from anomalib.models.image.dinomaly.components import dinov2_loader as dl

    route_dinov2_weights_through_core()
    _never_download(monkeypatch, dl)

    def _missing(cls, name, *args, **kwargs):
        raise ModelWeightsMissingError(
            f"'{name}' is not in the model cache. Provision it with: "
            f"uv run download-model download {name}"
        )

    monkeypatch.setattr(ModelWeights, "materialize", classmethod(_missing))

    loader = dl.DinoV2Loader(cache_dir=str(tmp_path / "cache"))
    with pytest.raises(
        ModelWeightsMissingError, match="download-model download dinov2_vitb14_reg4"
    ):
        loader._download_weights("dinov2_reg", "base", 14)


def test_routing_is_idempotent():
    from anomalib.models.image.dinomaly.components import dinov2_loader as dl

    route_dinov2_weights_through_core()
    first = dl.DinoV2Loader._download_weights
    route_dinov2_weights_through_core()
    assert dl.DinoV2Loader._download_weights is first
