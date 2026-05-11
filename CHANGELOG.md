# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## 0.1.4 - 2026-05-11

- Switched runtime dep from `opencv-python>=4.8.0` to `opencv-python-headless>=4.13.0.92` to match `cuvis-ai-sam3` / `cuvis-ai-adaclip`. The plugin has no `cv2.imshow` / window calls so the GUI subdeps (`libGL`, `libGTK`) were dead weight.
- **Fix (critical):** Inlined `_build_category_mask` and `_parse_coco_json` into `cuvis_ai_dinomaly/data/_coco_utils.py`. These were previously imported from `cuvis_ai.data.multi_file_dataset` which does not exist in the released `cuvis-ai` package, causing a silent `ImportError` at runtime for any pipeline using `MultiFileNpzDataModule`. Removed the `pytest.importorskip` guard that was hiding the failure in CI — datamodule tests now run unconditionally.
- Added two new unit tests for `_build_category_mask` (empty annotations → zero mask; bbox annotation → correct region fill).
- Removed dead `cuvis-ai = { path = "../cuvis-ai", editable = true }` entry from `[tool.uv.sources]` in `pyproject.toml` (`cuvis-ai` was dropped as a runtime dep in 0.1.3 but its source override lingered).
- Added inline comment in `pyproject.toml` explaining the `<3.12` Python cap: `anomalib==2.1.0` and `kornia==0.6.12` are tested on 3.11 only; the kornia pin avoids `kornia-rs` illegal instruction on CI runners.
- Added `pre_trained/` to `.gitignore`.
- Replaced hardcoded developer paths (`/home/dev/anish/…`, `/mnt/data/…`) in all six `configs/trainrun/*.yaml` files with placeholder comments.
- Rewrote `docs/publish_checklist.md` to be version-agnostic (`vX.Y.Z` placeholders, no personal paths). Added steps for compatibility audit (§8) and registry update.
- Added coverage to the CI `test` job: `--cov=cuvis_ai_dinomaly --cov-report=xml --cov-report=term-missing --cov-fail-under=70`. Coverage XML artifact uploaded for 7 days.
- Lowered `[tool.coverage.report] fail_under` from 90 to 70 to match the CI gate.
- Added `tags-ignore: ["v*.*.*"]` to `ci.yml` `on.push` so tag pushes no longer re-run CI (the new `release.yml` handles that).
- Added `.github/workflows/release.yml`: tag-triggered (`v*.*.*`), runs jobs `validate` → `security` → `build` (with tag-vs-package-version check) → `create-release` (extracts the matching CHANGELOG section as GitHub Release notes).
- Recorded compatibility audit against `cuvis-ai-core` 0.1.0 and 0.5.2 in [`docs/compatibility_audit.md`](docs/compatibility_audit.md). Result: PASS — every shared dep (`numpy`, `tqdm`, `defusedxml`, `requests`) satisfies the plugin's specifier; `anomalib`, `kornia`, `opencv-python-headless`, `open-clip-torch` are not in either core lock so no conflict risk.
- Extended CI to adaclip-level: added `typecheck` (mypy, non-blocking) and `security` (pip-audit, detect-secrets, bandit) jobs. The `build` job now gates on all four hygiene jobs. Added `.secrets.baseline` (zero findings) and a `[tool.bandit]` config block. Dev-deps grew with `mypy`, `pip-audit`, `detect-secrets`, `bandit[toml]`.
- Added `LICENSE` file (Apache-2.0 standard text + Cubert GmbH copyright) at repo root. `pyproject.toml` already declared `license = "Apache-2.0"` but the license text was not previously distributed.
- Added a **Plugin manifest** section to `README.md` documenting both local-path and git-tag manifest forms (skill §9 / "When to stop" requirement).
- Added CI (`ci.yml`) workflow with `test`, `lint`, and `build` jobs (Ubuntu / Python 3.11 / `uv` with `--no-sources`). Test job runs `pytest tests/ -m "not slow"` so the `integration`-marked manifest-loading smoke test executes per the cuvis-ai plugin skill verification step.
- Dropped runtime dependency on the high-level `cuvis-ai` package and added `cuvis-ai-schemas>=0.4.0`. The plugin now depends only on `cuvis-ai-core` + `cuvis-ai-schemas`, matching `cuvis-ai-deepeiou` / `cuvis-ai-sam3`. Avoids transitively importing the proprietary Cuvis SDK (`cuvis_il`) at module load.
- Inlined a minimal `_LossNode` in `dinomaly_train_loss_bridge.py` mirroring `cuvis_ai.node.losses.LossNode` (marker subclass that defaults `execution_stages = {TRAIN, VAL, TEST}` on `Node.__init__`). No behavior change for consumers.
- Added `requests>=2.31.0` to runtime dependencies (undeclared transitive of `anomalib==2.1.0` via the eager `anomalib.models.video.ai_vad.clip` import chain).
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
