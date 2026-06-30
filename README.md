# cuvis-ai-dinomaly

[![CI](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly/actions/workflows/ci.yml/badge.svg)](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

A **[cuvis.ai](https://github.com/cubert-hyperspectral/cuvis-ai) plugin** for [Dinomaly](https://github.com/guojiajeremy/Dinomaly) (CVPR 2025), a DINOv2-based multi-class unsupervised anomaly detection method, integrated via Intel **[Anomalib](https://github.com/open-edge-platform/anomalib)** `DinomalyModel`.

> **Note**: For upstream attribution and Anomalib integration details, see [`cuvis_ai_dinomaly/upstream/README.md`](cuvis_ai_dinomaly/upstream/README.md).

## Installation

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) (Python dependency manager)
- GPU recommended — DINOv2 encoder weights are downloaded on first run

> **Note**: This plugin depends on `cuvis-ai-core` and `cuvis-ai-schemas` only. It does **not** depend on the high-level `cuvis-ai` package, which avoids transitively importing the proprietary Cuvis SDK at module load.

### Setup

```bash
git clone https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly.git
cd cuvis-ai-dinomaly
uv sync --extra dev
```

For a clean install that resolves all dependencies from public indexes only (ignores local `[tool.uv.sources]` overrides — matches CI):

```bash
uv sync --no-sources --extra dev
```

## Nodes

### `DinomalyDetector`

Pixel-level anomaly detection wrapping Anomalib's `DinomalyModel`. The DINOv2 encoder stays frozen; only the bottleneck and decoder are trained.

**Inputs:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| `rgb_image` | `torch.float32` | `[B, H, W, 3]` | RGB or false-colour image (0–1 or 0–255) |

**Outputs:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| `scores` | `torch.float32` | `[B, H, W, 1]` | Pixel-wise anomaly heatmap |
| `anomaly_score` | `torch.float32` | `[B]` | Image-level anomaly score |
| `training_loss` | `torch.float32` | `()` | Scalar Dinomaly loss (train/val/test, optional) |

**Key Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder_name` | str | `"dinov2reg_vit_base_14"` | DINOv2 encoder variant |
| `image_size` | int | `448` | Resize target (height and width) |
| `crop_size` | int | `392` | Center-crop size after resize |
| `use_center_crop` | bool | `True` | Apply center crop; set `False` to skip (no-crop Anomalib setup) |
| `bottleneck_dropout` | float | `0.2` | Dropout rate in the bottleneck |
| `decoder_depth` | int | `8` | Number of decoder layers |

### `DinomalyTrainLossBridge`

Adapter node that maps `training_loss` → `loss` for the `GradientTrainer` `loss_nodes` contract. Include it in `loss_nodes` alongside `DinomalyDetector` in `unfreeze_nodes`.

## Plugin manifest

Drop one of these into your `plugins.yaml` and load it with `NodeRegistry().register_plugin("plugins.yaml")` (core ≥ 0.10; the old `load_plugins` was removed). The manifest uses the capabilities shape: a top-level `name`, a source (`path`, or `repo` + `tag`), and a `capabilities` list of `class_name` entries.

**Local path** (development; iterate alongside the consumer pipeline):

```yaml
name: dinomaly
path: "../cuvis-ai-dinomaly"
capabilities:
  - class_name: cuvis_ai_dinomaly.node.dinomaly_detector.DinomalyDetector
  - class_name: cuvis_ai_dinomaly.node.dinomaly_train_loss_bridge.DinomalyTrainLossBridge
  - class_name: cuvis_ai_dinomaly.node.auroc_metrics.AnomalyAUROCMetrics
```

**Git tag** (reproducible install from a tagged release):

```yaml
name: dinomaly
repo: "https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly.git"
tag: "v0.1.4"
capabilities:
  - class_name: cuvis_ai_dinomaly.node.dinomaly_detector.DinomalyDetector
  - class_name: cuvis_ai_dinomaly.node.dinomaly_train_loss_bridge.DinomalyTrainLossBridge
  - class_name: cuvis_ai_dinomaly.node.auroc_metrics.AnomalyAUROCMetrics
```

A ready-to-use local-path manifest is committed at [`examples/plugins.yaml`](examples/plugins.yaml).

## Examples

Training scripts are in [`examples/`](examples/). Each pairs with a matching Hydra config in [`configs/trainrun/`](configs/trainrun/).

| Script | Description |
|--------|-------------|
| `train_dinomaly_rgb_multifile.py` | Train on false-RGB from NPZ files |
| `train_dinomaly_cir_multifile.py` | Train on CIR (NIR/Red/Green) band mapping |
| `train_dinomaly_concrete_multifile.py` | Train with a fixed concrete band selector |
| `train_dinomaly_concrete_selector_multifile.py` | Train with learnable band selector |
| `train_dinomaly_concrete_joint_multifile.py` | Joint training of selector + detector |
| `train_dinomaly_rgb_frozen_adaclip_bands_multifile.py` | Fixed bands from a frozen AdaCLIP selector |
| `run_saved_dinomaly_pipeline_test_npz.py` | Run inference on saved pipeline |
| `export_dinomaly_multifile_pipeline_from_ckpt.py` | Export pipeline from checkpoint |

Quick start (edit `splits_csv` in the config first):

```bash
uv run python examples/train_dinomaly_rgb_multifile.py
```

## cuvis-ai configs

The `configs/trainrun/` directory in this repo contains Hydra train-run configs. In the main **cuvis-ai** repo, manifests assume `cuvis-ai-dinomaly` is a sibling folder. If your layout differs, edit `path:` in `configs/plugins/dinomaly.yaml`.

Gradient training uses `GradientTrainer` with `loss_nodes: [dinomaly_train_loss]` and `unfreeze_nodes: [dinomaly_detector]`.

### Optimizer parity (Anomalib vs cuvis-ai)

Anomalib's `Dinomaly` Lightning module uses **StableAdamW** with a warm cosine schedule. `cuvis-ai-core` registers **Adam**, **AdamW**, and **SGD** only. The example trainrun configs use **AdamW** (`lr=2e-3`, `weight_decay=1e-4`) to approximate Dinomaly defaults. Single-step loss semantics match Anomalib's `DinomalyModel` (verified in tests).

## Development

```bash
# Run fast test suite (matches CI)
uv run pytest tests/ -m "not slow"

# Run parity tests vs raw DinomalyModel (downloads weights, slow)
uv run pytest tests/test_parity.py -m slow

# Lint and format
uv run ruff format cuvis_ai_dinomaly tests examples
uv run ruff check cuvis_ai_dinomaly tests examples

# Build wheel
uv build --no-sources
```

Set `CUVIS_DINOMALY_SKIP_SLOW=1` to skip slow tests when running locally.

## Compatibility

- **Python**: 3.11 (`<3.12` required — `anomalib==2.1.0` and `kornia==0.6.12` are tested on 3.11 only)
- **PyTorch**: CUDA build via `pytorch-cu128` index (see `[tool.uv.sources]`)
- Audited against `cuvis-ai-core` `v0.1.0` (declared floor) and `v0.5.2` (latest released) — PASS. See [`docs/compatibility_audit.md`](docs/compatibility_audit.md).

## Citation

If you use this plugin in your work, please cite the original Dinomaly paper:

```bibtex
@inproceedings{gu2025dinomaly,
  title     = {Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection},
  author    = {Gu, Jia-Li and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```

Also cite Anomalib if you use it through this plugin:

```bibtex
@inproceedings{akcay2022anomalib,
  title     = {Anomalib: A Deep Learning Library for Anomaly Detection},
  author    = {Akcay, Samet and others},
  booktitle = {2022 IEEE International Conference on Image Processing (ICIP)},
  year      = {2022}
}
```

## License

Apache-2.0 — full text in [`LICENSE`](LICENSE). Aligns with Anomalib and the original Dinomaly repository. Attribute the original paper and repositories in derivative work.

## Acknowledgments

- Original Dinomaly (CVPR 2025): [guojiajeremy/Dinomaly](https://github.com/guojiajeremy/Dinomaly)
- Anomalib integration: [open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)
- cuvis.ai framework: [cubert-hyperspectral/cuvis-ai](https://github.com/cubert-hyperspectral/cuvis-ai)
