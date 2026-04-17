"""Train Dinomaly on multi-file data with a learnable Concrete channel selector.

This mirrors the RGB/CIR scripts but replaces fixed/CIR wavelength selection with
`ConcreteChannelMixer` (learned spectral projection to 3 channels).

Usage (from cuvis-ai-dinomaly repo root):
    uv run python examples/train_dinomaly_concrete_multifile.py
    uv run python examples/train_dinomaly_concrete_multifile.py data.splits_csv=lentils_splits.csv

Full 50-epoch NPZ run (best-ckpt val/test + export), mirroring CIR workflow:
    uv run python examples/train_dinomaly_concrete_multifile.py \
      output_dir=/mnt/data/cuvis_ai_outputs/dinomaly_concrete_npz_50ep_w0 \
      training.trainer.max_epochs=50 \
      data.splits_csv=/home/dev/anish/cuvis-ai-dinomaly/diagnostics/lentils_splits_npz_full.csv \
      data.num_workers=0 data.persistent_workers=false \
      eval_mode=best
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
    config_name="trainrun/dinomaly_multifile_concrete",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    run_dinomaly_multifile_training(
        cfg,
        band_mode="concrete",
        run_title="=== Dinomaly + MultiFile (Concrete selector) ===",
    )


if __name__ == "__main__":
    main()

