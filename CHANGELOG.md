# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Added `LICENSE` file (Apache-2.0 standard text + Cubert GmbH copyright) at repo root. `pyproject.toml` already declared `license = "Apache-2.0"` but the license text was not previously distributed.
- Added a **Plugin manifest** section to `README.md` documenting both local-path and git-tag manifest forms (skill §9 / "When to stop" requirement).
- Recorded compatibility audit against `cuvis-ai-core` 0.1.0 and 0.5.2 in [`docs/compatibility_audit.md`](docs/compatibility_audit.md). Result: PASS — every shared dep (`numpy`, `tqdm`, `defusedxml`, `requests`) satisfies the plugin's specifier; `anomalib`, `kornia`, `opencv-python`, `open-clip-torch` are not in either core lock so no conflict risk.
- Added CI (`ci.yml`) workflow with `test`, `lint`, and `build` jobs (Ubuntu / Python 3.11 / `uv` with `--no-sources`). Test job runs `pytest tests/ -m "not slow"` so the `integration`-marked manifest-loading smoke test executes per the cuvis-ai plugin skill verification step.
- Dropped runtime dependency on the high-level `cuvis-ai` package and added `cuvis-ai-schemas>=0.4.0`. The plugin now depends only on `cuvis-ai-core` + `cuvis-ai-schemas`, matching `cuvis-ai-deepeiou` / `cuvis-ai-sam3`. Avoids transitively importing the proprietary Cuvis SDK (`cuvis_il`) at module load.
- Inlined a minimal `_LossNode` in `dinomaly_train_loss_bridge.py` mirroring `cuvis_ai.node.losses.LossNode` (marker subclass that defaults `execution_stages = {TRAIN, VAL, TEST}` on `Node.__init__`). No behavior change for consumers.
- Added `requests>=2.31.0` to runtime dependencies (undeclared transitive of `anomalib==2.1.0` via the eager `anomalib.models.video.ai_vad.clip` import chain).
- Added `pytest.importorskip` guards at the top of `tests/test_parity.py` and `tests/test_npz_datamodule.py` so collection skips cleanly when their respective heavy deps aren't fully resolved.
- Replaced `tests/test_parity_markers.py` `importlib.spec_from_file_location` + `exec_module` with `ast.parse`-based static inspection so the marker-checks no longer re-execute `test_parity.py` at collection time.
- Ran `ruff format` over `cuvis_ai_dinomaly/`, `tests/`, `examples/` and applied `ruff check --fix --unsafe-fixes` (4 import sorts + 5 `dict()` → `{}` rewrites in example training scripts).

## 0.1.3 - 2026-05-04

- Added `open-clip-torch>=2.24.0` runtime dependency. Anomalib model modules import WinCLIP / OpenCLIP symbols during package initialization.

## 0.1.2 - 2026-05-04

- Pinned `kornia==0.6.12` to avoid the resolver selecting builds that pull `kornia-rs` and crash with illegal instruction at runtime.

## 0.1.1 - 2026-05-04

- Pinned `anomalib==2.1.0` to avoid importing `kornia-rs`, which crashes with illegal instruction in the plugin runtime.

## 0.1.0 - 2026-04-17

- Added `cuvis_ai_dinomaly` plugin package with `DinomalyDetector` and `DinomalyTrainLossBridge` node classes.
- Added plugin scaffolding with `pyproject.toml`, LICENSE attribution (Apache-2.0), and README.
- Added Anomalib `DinomalyModel` wrapper with frozen DINOv2 encoder and trainable bottleneck + decoder.
- Added Hydra-based multifile training examples (RGB, CIR, concrete, joint, selector, frozen-AdaCLIP-bands).
- Added `examples/plugins.yaml` local-path manifest exposing both nodes.
- Added pytest suite with detector-forward, loss-bridge, port-contract, preprocess, manifest-load, and parity tests (slow tests gated behind `CUVIS_DINOMALY_SKIP_SLOW`).
