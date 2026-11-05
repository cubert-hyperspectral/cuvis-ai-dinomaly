"""Train Dinomaly on multi-file .cu3s data with RGB (fixed-wavelength) false color.

Wires :class:`DinomalyTrainLossBridge` in the pipeline graph and passes it to
:class:`GradientTrainer` as ``loss_nodes`` (not only via YAML).

Usage (from cuvis-ai-dinomaly repo root):
    uv run python examples/train_dinomaly_rgb_multifile.py
    uv run python examples/train_dinomaly_rgb_multifile.py \
      data.universe_csv=<lentils>/universe.csv data.splits_json=<lentils>/splits/dinomaly.json
    uv run python examples/train_dinomaly_rgb_multifile.py dinomaly.use_center_crop=false

<lentils> is the HF lentils download (the cu3s recordings plus universe.csv and splits/).

Full 50-epoch run (best-ckpt val/test + export), mirroring CIR workflow:
    uv run python examples/train_dinomaly_rgb_multifile.py \
      output_dir=/mnt/data/cuvis_ai_outputs/dinomaly_rgb_50ep \
      training.max_epochs=50 \
      data.universe_csv=<lentils>/universe.csv data.splits_json=<lentils>/splits/dinomaly.json \
      eval_mode=best

The data.reader defaults fit an 8 GB GPU; on a 24 GB GPU data.reader.max_open_sessions=6 makes
an epoch about twice as fast (examples/LENTILS_TRAINING.md, section 4b).
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
    config_name="trainrun/dinomaly_multifile_rgb",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    run_dinomaly_multifile_training(
        cfg,
        band_mode="rgb",
        run_title="=== Dinomaly + MultiFileCu3s (RGB fixed wavelengths) ===",
    )


if __name__ == "__main__":
    main()
