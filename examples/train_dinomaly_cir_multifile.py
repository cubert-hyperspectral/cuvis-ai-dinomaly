"""Train Dinomaly on multi-file .cu3s data with CIR (NIR/R/G) false color.

Same graph pattern as ``train_dinomaly_rgb_multifile.py`` but uses :class:`CIRSelector`.

Usage (from cuvis-ai-dinomaly repo root):
    uv run python examples/train_dinomaly_cir_multifile.py
    uv run python examples/train_dinomaly_cir_multifile.py data.splits_csv=lentils_splits.csv
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
    config_name="trainrun/dinomaly_multifile_cir",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    run_dinomaly_multifile_training(
        cfg,
        band_mode="cir",
        run_title="=== Dinomaly + MultiFileCu3s (CIR) ===",
    )


if __name__ == "__main__":
    main()
