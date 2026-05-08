# cuvis-ai-dinomaly

![CI](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly/actions/workflows/ci.yml/badge.svg)

[Dinomaly](https://github.com/guojiajeremy/Dinomaly) anomaly detection as a **cuvis.ai plugin**, implemented with Intel **[Anomalib](https://github.com/open-edge-platform/anomalib)** `DinomalyModel`.

## Requirements

- Python 3.11
- `cuvis-ai-core>=0.1.0` and `cuvis-ai-schemas>=0.4.0` — the plugin does **not** depend on the high-level `cuvis-ai` package (avoids transitively importing the proprietary Cuvis SDK at module load)
- GPU recommended (DINOv2 backbone is downloaded on first run)

## Install

For development from a local checkout:

```bash
cd cuvis-ai-dinomaly
uv sync --extra dev
```

For a clean install that resolves everything from public indexes (ignores the editable `[tool.uv.sources]` overrides):

```bash
uv sync --no-sources --extra dev
```

### Using the plugin from the main `cuvis-ai` repo

When the plugin is loaded by path from a sibling `cuvis-ai` checkout, all runtime deps (`anomalib`, `kornia`, `opencv-python`, `open-clip-torch`, `requests`, …) come along automatically via the manifest install — no manual `uv pip install` step needed.

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
uv run pytest tests/ -m "not slow"          # fast suite (matches CI)
uv run pytest tests/test_parity.py -m slow  # parity vs raw DinomalyModel (downloads weights)
```

Set `CUVIS_DINOMALY_SKIP_SLOW=1` to skip slow tests when running locally. CI runs only the fast suite (`-m "not slow"`); the `integration`-marked manifest-loading smoke test is included.

## Compatibility

Audited per the cuvis-ai plugin skill §8 against `cuvis-ai-core` `v0.1.0` (declared floor) and `v0.5.2` (latest released): every shared dep satisfies the plugin's specifier. See [`docs/compatibility_audit.md`](docs/compatibility_audit.md) for the per-dep verdict.

## License

Apache-2.0 — full text in [`LICENSE`](LICENSE). Aligns with Anomalib / typical Dinomaly redistribution. Attribute the original paper and repos in derivative work.
