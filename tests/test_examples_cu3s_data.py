"""The lentils examples build their data module from a trainrun's ``data`` section."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

omegaconf = pytest.importorskip("omegaconf")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class _FakeDataModule:
    """Records what the examples hand to MultiCu3sDataModule (the examples extra is not in dev)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.stages: list[str] = []

    def setup(self, stage: str) -> None:
        self.stages.append(stage)


@pytest.fixture
def cu3s_data(monkeypatch):
    fake = ModuleType("cuvis_ai_dataloader.data")
    fake.MultiCu3sDataModule = _FakeDataModule  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cuvis_ai_dataloader.data", fake)
    monkeypatch.syspath_prepend(str(EXAMPLES))
    sys.modules.pop("cu3s_data", None)
    import cu3s_data

    yield cu3s_data
    sys.modules.pop("cu3s_data", None)


def _data(**overrides):
    data = {
        "data_module": "cu3s_multi",
        "universe_csv": "lentils/universe.csv",
        "batch_size": 1,
        "processing_mode": "Reflectance",
        "num_workers": 0,
        "reader": {"max_open_sessions": 3, "read_ahead": 1, "cuda_cubes": False, "sdk_cuda": True},
    }
    data.update(overrides)
    return omegaconf.OmegaConf.create(data)


def test_reader_options_and_splits_reach_the_data_module(cu3s_data, monkeypatch):
    monkeypatch.setattr(cu3s_data, "load_splits", lambda path: f"splits from {path}")
    dm = cu3s_data.build_cu3s_datamodule(_data(splits_json="lentils/splits/dinomaly.json"))
    assert dm.kwargs == {
        "universe_csv": "lentils/universe.csv",
        "splits": "splits from lentils/splits/dinomaly.json",
        "batch_size": 1,
        "num_workers": 0,
        "processing_mode": "Reflectance",
        "max_open_sessions": 3,
        "read_ahead": 1,
        "cuda_cubes": False,
        "sdk_cuda": True,
    }
    assert dm.stages == ["fit"]


def test_without_splits_json_or_reader_the_defaults_apply(cu3s_data):
    data = _data()
    del data["reader"]
    dm = cu3s_data.build_cu3s_datamodule(data)
    assert dm.kwargs["splits"] is None
    assert "max_open_sessions" not in dm.kwargs
