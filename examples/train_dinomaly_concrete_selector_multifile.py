"""Train only Concrete selector for Dinomaly with selector regularization.

This variant keeps Dinomaly frozen and optimizes only `concrete_selector` using:
- Dinomaly train loss bridge (drives selector through frozen Dinomaly)
- DistinctnessLoss on selector weights (prevents channel collapse)

Usage (from cuvis-ai-dinomaly repo root):
    uv run python examples/train_dinomaly_concrete_selector_multifile.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from dinomaly_multifile_train_common import run_dinomaly_multifile_training


@hydra.main(
    config_path="../configs",
    config_name="trainrun/dinomaly_multifile_concrete_selector",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    run_dinomaly_multifile_training(
        cfg,
        band_mode="concrete",
        run_title="=== Dinomaly + MultiFile (Concrete selector-only + distinctness) ===",
    )


if __name__ == "__main__":
    main()
