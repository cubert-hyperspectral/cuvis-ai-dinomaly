# cuvis-ai-dinomaly

[Dinomaly](https://github.com/guojiajeremy/Dinomaly) anomaly detection as a **cuvis.ai plugin**, implemented with Intel **[Anomalib](https://github.com/open-edge-platform/anomalib)** `DinomalyModel`.

## Requirements

- Python 3.11
- `cuvis-ai` / `cuvis-ai-core` (see `pyproject.toml` `[tool.uv.sources]` for local vs Git)
- GPU recommended (DINOv2 backbone download on first run)

## Install

```bash
cd cuvis-ai-dinomaly
uv sync --extra dev
```

Adjust `pyproject.toml` paths for `cuvis-ai` and `cuvis-ai-core` on your machine.

### Using the plugin from the main `cuvis-ai` repo

Install the same runtime deps into the **cuvis-ai** virtualenv (plugin code is loaded by path):

```bash
cd /path/to/cuvis-ai
uv pip install "anomalib>=2.1,<3" "opencv-python>=4.8"
```

If `from cv2 import imread` fails, reinstall `opencv-python` (a broken `cv2` namespace package will prevent Anomalib from importing).

## Nodes

| Class | Role |
|-------|------|
| `cuvis_ai_dinomaly.node.dinomaly_detector.DinomalyDetector` | RGB `[B,H,W,3]` in → `scores` `[B,H,W,1]`, `anomaly_score` `[B]`, `training_loss` (train/val/test) |
| `cuvis_ai_dinomaly.node.dinomaly_train_loss_bridge.DinomalyTrainLossBridge` | Maps `training_loss` → `loss` for `GradientTrainer` `loss_nodes` |

Encoder weights are frozen; only **bottleneck + decoder** train (same as Anomalib’s Lightning module).

## Plugin manifest

Drop one of these into your `plugins.yaml` and load with `NodeRegistry.load_plugins(...)`.

**Local path** (development; iterate alongside the consumer pipeline):

```yaml
plugins:
  dinomaly:
    path: "../cuvis-ai-dinomaly"
    provides:
      - cuvis_ai_dinomaly.node.dinomaly_detector.DinomalyDetector
      - cuvis_ai_dinomaly.node.dinomaly_train_loss_bridge.DinomalyTrainLossBridge
```

**Git tag** (reproducible install from a tagged release):

```yaml
plugins:
  dinomaly:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly.git"
    tag: "v0.1.3"
    provides:
      - cuvis_ai_dinomaly.node.dinomaly_detector.DinomalyDetector
      - cuvis_ai_dinomaly.node.dinomaly_train_loss_bridge.DinomalyTrainLossBridge
```

A ready-to-use local-path manifest is committed at [`examples/plugins.yaml`](examples/plugins.yaml).

## cuvis-ai configs

In the main **cuvis-ai** repo, manifests assume **`cuvis-ai-dinomaly` is a sibling folder** of `cuvis-ai` (e.g. `anish/cuvis-ai` and `anish/cuvis-ai-dinomaly`). If your layout differs, edit `path:` in `configs/plugins/dinomaly.yaml` and `configs/plugins/registry.yaml`.

- Pipeline: `configs/pipeline/anomaly/dinomaly/dinomaly_baseline.yaml`
- Train run: `configs/trainrun/dinomaly_baseline.yaml`
- Plugin manifest: `configs/plugins/dinomaly.yaml`

Restore the pipeline (example path on Linux):

```bash
cd /path/to/cuvis-ai
uv run restore-pipeline \
  --pipeline-path configs/pipeline/anomaly/dinomaly/dinomaly_baseline.yaml \
  --plugins-path configs/plugins/dinomaly.yaml
```

Gradient training uses `GradientTrainer` with `loss_nodes: [dinomaly_train_loss]` and `unfreeze_nodes: [dinomaly_detector]`. See `examples/advanced/deep_svdd_gradient_training.py` in cuvis-ai for the same orchestration pattern.

`restore-trainrun` now forwards `training.scheduler` into `GradientTrainer` (cuvis-ai-core); use Hydra overrides if you need a different schedule.

## Optimizer parity (Anomalib vs cuvis-ai)

Anomalib’s `Dinomaly` Lightning module uses **StableAdamW** and a custom **warm cosine** schedule. `cuvis-ai-core` currently registers **Adam**, **AdamW**, and **SGD** only (`cuvis_ai_core.training.optimizer_registry`). The example trainrun uses **AdamW** with `lr=2e-3` and `weight_decay=1e-4` to approximate Dinomaly defaults. Multi-step learning curves may still differ from `anomalib.engine.Engine.fit`; single-step loss semantics match Anomalib’s `DinomalyModel` (see tests).

## Tests

```bash
uv run pytest tests/ -m "not slow"   # preprocessing + loss bridge
uv run pytest tests/test_parity.py -m slow   # parity vs raw DinomalyModel (downloads weights)
```

Set `CUVIS_DINOMALY_SKIP_SLOW=1` to skip slow tests in CI.

## License

Apache-2.0 (aligns with Anomalib / typical Dinomaly redistribution). Attribute the original paper and repos in derivative work.
