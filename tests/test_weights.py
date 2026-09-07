"""The plugin's weight declaration (``WEIGHTS``) and the loader table derived from it.

No network and no checkpoint: the declaration is data and the registry is in-process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cuvis_ai_core.data.model_weights import ModelWeights
from cuvis_ai_schemas.plugin import PluginWeightEntry

import cuvis_ai_dinomaly
import cuvis_ai_dinomaly.weights as weights_mod
from cuvis_ai_dinomaly.node._dinov2_cache import MIRRORED_DINOV2_WEIGHTS
from cuvis_ai_dinomaly.weights import PLUGIN_NAME, WEIGHTS


def test_declares_the_dinov2_backbone_with_full_pins() -> None:
    assert [entry.name for entry in WEIGHTS] == ["dinov2_vitb14_reg4"]
    (entry,) = WEIGHTS
    assert isinstance(entry, PluginWeightEntry)
    assert entry.repo_id == "cubert-gmbh/dinov2"
    assert entry.filename == "dinov2_vitb14_reg4_pretrain.pth"
    assert len(entry.revision) == 40 and len(entry.sha256) == 64
    assert entry.size_bytes == 346_393_545
    assert entry.selected_by is None and entry.default is False
    assert entry.explicit_path_hparams == []
    assert entry.license == "Apache-2.0" and entry.license_file == "LICENSE"
    assert "Backbone" in entry.used_for and entry.summary and entry.description


def test_register_called_at_import() -> None:
    assert PLUGIN_NAME == "dinomaly"
    row = ModelWeights.get("dinov2_vitb14_reg4")
    assert row.plugin == PLUGIN_NAME
    assert row.source == "plugin"
    assert row.entry == WEIGHTS[0]
    assert cuvis_ai_dinomaly.WEIGHTS is WEIGHTS


def test_loader_filename_table_is_derived_from_the_declaration() -> None:
    assert MIRRORED_DINOV2_WEIGHTS == {entry.filename: entry.name for entry in WEIGHTS}
    assert MIRRORED_DINOV2_WEIGHTS == {"dinov2_vitb14_reg4_pretrain.pth": "dinov2_vitb14_reg4"}


def test_trained_pipelines_stay_core_owned() -> None:
    """The Cubert-trained Dinomaly pipelines are core's rows; the plugin declares only the backbone."""
    declared = {entry.name for entry in WEIGHTS}
    assert not any(name.startswith("dinomaly_") for name in declared)
    for row in ModelWeights.rows():
        if row.source == "dict":
            assert row.entry.name not in declared


def test_weights_module_is_side_effect_free() -> None:
    """The module declares only: loading it alone must not import torch, anomalib, core or the plugin."""
    path = Path(weights_mod.__file__)
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('weights_probe', {str(path)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "assert len(mod.WEIGHTS) == 1\n"
        "for heavy in ('torch', 'anomalib', 'cuvis_ai_core', 'cuvis_ai_dinomaly'):\n"
        "    assert heavy not in sys.modules, heavy\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180, check=False
    )
    assert result.returncode == 0, result.stderr
