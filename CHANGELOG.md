# Changelog

## 0.6.4 - 2026-09-04

- The streaming metric base (`ValNormalAnomalyMean`, the AUROC metrics) and `DinomalyTrainLossBridge` declare their execution stages on the class (`EXECUTION_STAGES`, cuvis-ai-core 0.14.1) instead of passing `execution_stages=` to the constructor; `consume_base_kwargs` is no longer used. The example distinctness loss follows. Floors `cuvis-ai-core>=0.14.1`.

## 0.6.3 - 2026-08-31

- `DinomalyDetector` now aligns the returned anomaly map to the input pixel grid (new `align_map_to_input` flag, default on). anomalib's `DinomalyModel` upsamples the low-resolution patch anomaly map to `image_size` with `align_corners=True`, while every other resize on the node's path (the preprocessing `Resize` and the final scores-to-native `F.interpolate`) is area-based; the convention mismatch displaced the returned map radially outward: zero at the image centre, growing toward the edges (12.3 px removed at 457 px radius on the lentils champion checkpoint). A scoped context manager forces `align_corners=False` in anomalib's internal upsample for the duration of the model call; it changes only where scores land, not their values. Because every returned map shifts slightly, a deployed decider `image_threshold` tuned on the old maps is worth a re-check; set `align_map_to_input: false` to reproduce the previous behaviour. The proxy is a single module-level instance (not per-call) so `torch.compile` guard identity stays stable under `compile_mode` / `fast_inference=True`, and the flag is forwarded into the node's hparams so the opt-out survives pipeline save/restore.

## 0.6.2 - 2026-08-26

- `AnomalyAUROCMetrics` now opts into the trainer's pooled epoch-end reduction: it declares `POOLED_METRIC_NAMES = {"auroc_pixel", "auroc_image"}` and exposes `pooled_metrics()` returning the live `BinaryAUROC` accumulators, so `GradientTrainer` (cuvis-ai-core >= 0.10.1) skips their per-batch float logging and logs the metric objects with `on_epoch=True`, making the reported epoch value one exact pooled `compute()` instead of the batch-size-sensitive mean of per-batch running values (badly biased at `batch_size=1`, where early single-class batches contribute 0.0; measured ~0.82 reported vs 0.994 true pixel AUROC on the lentils run). The per-batch running values stay on the `metrics` port for live monitoring, and the 0.3.0 "monitoring-only" caveat no longer applies to the trainer-reported value. `PerClassAnomalyAUROC` is unchanged (read via `compute()`, not the trainer table).

## 0.6.1 - 2026-08-20

- Removed the dead `[tool.uv.sources]` / `[[tool.uv.index]]` torch cu128 configuration: torch is not a direct dependency of this package and the committed lock (generated with --no-sources) resolves it from PyPI, so the tables had no effect anywhere. Composed child environments receive the host-mirrored torch build from cuvis-ai-core >= 0.12.1.

## 0.6.0 - 2026-08-20

- Adopted the dataloader's unified `universe.csv` vocabulary: the npz universe input's `path` column is renamed `materialized_path`, and the cu3s backend takes `universe_csv` (was `splits_csv`), matching the dataloader change that gives `cu3s_multi` and `npz_multi` one shared column set. `notebooks/lentils_anomaly/utils.py` reads `materialized_path`; REPRODUCTION.md and LENTILS_TRAINING.md document the shared vocabulary.
- Backend selection now goes through `data.data_module` (`npz_multi` default, or `cu3s_multi`): both backends read `universe_csv`, so the old discriminator (npz if `universe_csv` is set, else cu3s via `splits_csv`) no longer works. The multifile example scripts and `run_saved_dinomaly_pipeline_test_npz.py` branch on `--data-module`; the 6 `configs/trainrun/*.yaml` set `data_module: npz_multi` and document the cu3s alternative; `profile_saved_dinomaly_pipeline_cu3s.py` takes `--universe-csv` (was `--splits-csv`).
- Repointed the four `notebooks/lentils_anomaly/` tutorials (rgb, cir, adaclip_bands, inference) to `convert_universe`: they now download the dataset and call `cuvis_ai_dataloader.data.npz_converter.convert_universe(universe.csv, splits_json=splits/dinomaly.json)` instead of `convert_split_manifest(splits_dinomaly.csv)`, because the HF dataset now ships a `universe.csv` plus `splits/*.json` selectors rather than the rich cu3s split CSV. The `lentils_adaclip_bands` cell deriving the AdaCLIP wavelengths reads `materialized_path`; REPRODUCTION.md, LENTILS_TRAINING.md, and `utils.py` describe the `convert_universe` flow.
- Added `ValNormalAnomalyMean` (landed via PR 12 without a changelog entry): a metric node keeping a running mean of the image-level `anomaly_score` per (stage, epoch) on VAL/TEST, a label-free monitoring signal for normals-only training where mask metrics are degenerate. Registered in the plugin manifest.
- Moved the plugin's own manifest from `examples/plugins.yaml` to `configs/plugins/dinomaly.yaml`, next to the provisioned `cuvis_ai_dataloader.yaml`, so `configs/plugins/` works as a `--plugins-dir`. Updated every reference (example scripts, tests, notebooks, README, publish checklist, LENTILS_TRAINING.md, REPRODUCTION.md) and refreshed the README manifest snippets (current tag, all five capabilities).
- Extracted the shared (stage, epoch) reset bookkeeping into a `_StreamingMetric` base (new module `cuvis_ai_dinomaly/node/_streaming_metric.py`); `_StreamingBinnedAUROC` and `ValNormalAnomalyMean` both inherit it. `ValNormalAnomalyMean`'s docstring now states the normals-only input contract and the mean-of-running-means epoch reduction caveat.
- Rewrote the bedding tutorial notebooks onto the standard inline data provisioning: `PublicDatasets.download_dataset("industrial_fod_bedding")` then `convert_split_manifest` with `crop=(300, 300, 300, 300)` and `processing_mode=None` over the dataset's `splits.csv` and per-session COCO jsons. Deleted the bespoke HF wrappers, `prepare_bedding_data`, and the `BEDDING_DATA_SOURCE` module toggle from `notebooks/bedding_anomaly/utils.py`, which now holds only thin rendering helpers. Fixed `panel_frames` to read the universe's `materialized_path` column (was `path`) in both the bedding and lentils utils. The bedding inference notebook ships with outputs cleared pending a re-bake.
- Lentils notebook cleanup: dropped the dead `resolve_adaclip_wavelengths` and `per_class_pixel_auroc` helpers from `notebooks/lentils_anomaly/utils.py`, removed empty code cells from the three train tutorials, and repointed their hardcoded manifest path.
- Raised the dependency floors to the current release train: `cuvis-ai-core>=0.12.1`, `cuvis-ai-schemas>=0.9.0`, and in the `examples` extra `cuvis-ai>=0.12.0` and `cuvis-ai-dataloader>=0.5.0` (the release carrying `convert_universe` and the `materialized_path` universe vocabulary). The `examples` extra now declares `pandas`, `matplotlib`, and `huggingface-hub` explicitly; the provisioned dataloader manifest tag is bumped to `v0.5.0`.
- Restyled this changelog into the flat terse-bullet house style (facts unchanged).

## 0.5.0 - 2026-07-22

- Added `PerClassAnomalyAUROC`, a streaming one-vs-background per-class pixel AUROC node: for each non-background class it accumulates a torchmetrics `BinaryAUROC` (histogram thresholds) over that class's pixels versus background, reusing the same binned primitive as `AnomalyAUROCMetrics` through a shared `_StreamingBinnedAUROC` base, and exposes the whole-run values via `.compute()` (read off the node like the pixel/image AUROC). It consumes a `class_mask` port (the multi-class ground truth, new in cuvis-ai's `AnomalyDataNode`), so a pipeline can compute per-class AUROC as a node instead of a notebook-side loop. Registered in `examples/plugins.yaml`. The lentils inference tutorial attaches it to the loaded pipeline and reads its section 4 per-class AUROC off the node via `.compute()` (like the overall metrics), dropping the per-frame flatten loop; section 2 keeps only a slim `utils.panel_frames` extraction for the qualitative panels.
- `DinomalyDetector` now declares a node `category` and `tags` so it self-describes in the node catalog (metadata only, no behavior change).
- Renamed the npz universe input `index_csv` to `universe_csv`: the `MultiNpzDataModule` argument and the `data.index_csv` trainrun key are now `universe_csv` / `data.universe_csv`, pointing at a `universe.csv` whose columns are `source, index, path` (was `npz_path, source, image_id`). Updated the 6 `configs/trainrun/*.yaml`, the multifile example scripts, `run_saved_dinomaly_pipeline_test_npz.py` (`--universe-csv`), and the tutorial notebooks. Regenerate the npz `universe.csv` (the converter emits the new columns); needs the cuvis-ai-dataloader carrying the same rename.
- Adopted the flat `TrainingConfig` (needs `cuvis-ai-core>=0.11.0` / `cuvis-ai-schemas>=0.8.0`): `TrainerConfig` was folded into `TrainingConfig` upstream, so the nested `trainer:` block is gone from `configs/training/default.yaml` and every `configs/trainrun/*.yaml` (its `pytorch_lightning.Trainer` fields now sit flat under `training:`), the example scripts and tutorial notebooks build a single flat `TrainingConfig` and pass `training_config=` to `GradientTrainer` (was `trainer_config=` / `optimizer_config=`), and Hydra overrides use `training.<field>=...` instead of `training.trainer.<field>=...`.
- Migrated the tutorial notebooks to the selector split model: the lentils notebooks (rgb, cir, adaclip_bands, inference) and the bedding train notebook build a `universe.csv` plus a baked `splits.json` and load via `MultiNpzDataModule(splits=DataSplitConfig(splits_path=...), universe_csv=...)` instead of the removed `splits_csv=`. Each notebook provisions data inline with the cuvis-ai-core tools directly (`PublicDatasets.download_dataset(...)` then `convert_split_manifest(...)`), so a reader sees the real API rather than a notebook-local wrapper. The shared `notebooks/lentils_anomaly/utils.py` drops the old `(split, npz_path, image_id)` CSV helpers (`ensure_lentils_npz`, `resolve_splits_csv`, `subsample_splits_csv`, and the `LENTILS_DATA_SOURCE` toggle), reads universe `path` / `index` records, and resolves the AdaCLIP bands from the `universe.csv`; the generated lentils `splits.json` is identical to the dataset's shipped `splits/dinomaly.json`. Also fixed a stale `TensorBoardMonitorNode` import and a `params={"universe": ...}` key in the RGB notebook, and refreshed REPRODUCTION.md.
- Brought the CIR, AdaCLIP-bands, and bedding train tutorials onto the RGB reference conventions, mirroring `lentils_rgb_train_tutorial.ipynb`: framework training via `restore_trainrun` plus a flat `TrainRunConfig` (not hand-rolled `StatisticalTrainer` / `GradientTrainer`), a node-built preview video (`... -> MaskOverlayNode -> ToVideoNode`), the graph shown as a self-rendering bare `pipeline`, `AnomalyDataNode` in place of the deprecated `LentilsAnomalyDataNode`, INFO-level logging with a cuda/mps/cpu device fallback, a Colab bootstrap, and a per-node profiling pass; CIR and AdaCLIP swap only the front-end selector, AdaCLIP resolving its bands from the first NPZ. Polished the inference tutorials too: the lentils inference notebook reads its per-class AUROC off the node, and the bedding inference notebook was rewritten onto the same arc, loading the published pipeline (attaching the metric nodes it lacks plus `PerClassAnomalyAUROC`), running a `Predictor` pass over the val split, and reporting overall plus live 23-class per-class AUROC, dropping the earlier single-frame walkthrough and the speedup-recipe section. Both use `uv` prerequisites and drop dead env-var/script references and em-dashes. The published `cubert-gmbh/dinomaly-bedding-all6` pipeline YAML was re-pointed from the deprecated `LentilsAnomalyDataNode` to the canonical `AnomalyDataNode` (a behaviorally identical alias; the data node holds no weights).
- Floored `cuvis-ai-dataloader>=0.4.0` (was 0.3.0): 0.4.0 is the release whose `MultiNpzDataModule` emits the per-frame `class_mask` batch key `PerClassAnomalyAUROC` reads, so the per-class section no longer silently no-ops. Also dropped the dead `build_selector` helper from `notebooks/lentils_anomaly/utils.py` (the tutorials build their selectors inline).

## 0.4.1 - 2026-07-17

- Raised the `cuvis-ai-core` floor to `>=0.11.2` and `cuvis-ai-schemas` to `>=0.8.0`, matching the flat `TrainingConfig` the trainer already targets. Dependency floors only; no API change.

## 0.4.0 - 2026-07-01

- Moved the NPZ data layer to cuvis-ai-dataloader: deleted `cuvis_ai_dinomaly/data/` (`MultiFileNpzDataset` / `MultiFileNpzDataModule` and the local `_coco_utils` COCO helpers); the generic loader now lives upstream as `cuvis_ai_dataloader.data.MultiNpzDataModule` (`data_module_name: npz_multi`, cuvis-ai-dataloader 0.3.0+). The example scripts and the bedding train notebook import it from there; the batch contract (`cube`, `mask`, `wavelengths`, `mesu_index`) is unchanged. Dropped the never-hit `annotation_json`-to-mask fallback (masks are baked into the NPZ, so the ecosystem keeps a single COCO source in cuvis-ai-dataloader's `coco_labeler`) and the bedding-only `class_mask` batch key (read by no pipeline node).
- Added `cuvis-ai-dataloader>=0.3.0` to the `examples` extra and bumped the provisioned `configs/plugins/cuvis_ai_dataloader.yaml` manifest to `v0.3.0` (adds the `npz_multi` capability).

## 0.3.0 - 2026-06-30

- Added n-channel input: `DinomalyDetector(input_channels=N)` inflates the pretrained DINOv2 patch-embed Conv2d from 3 to N channels by duplicate-and-halve, preserving activation magnitude at init. The inflated stem stays frozen (anomalib runs the encoder under `torch.no_grad()`, so it receives no gradient); only the bottleneck and decoder train. Defaults to 3 for full backward compatibility.
- Added rectangular `image_size` / `crop_size`: `DinomalyDetector` now accepts an `int` (square, unchanged) or an `(h, w)` tuple. For non-square inputs it patches anomalib's square-grid reshape in `DinomalyModel.get_encoder_decoder_outputs`; the square path stays byte-identical. A hard version guard raises if the installed anomalib is outside the verified set, since the rectangular reshape copies anomalib internals verbatim.
- Added an optional `fast_inference` API: new `fast_inference` / `use_tf32` / `autocast_dtype` / `compile_mode` kwargs plus `warmup()` enable a validated TF32 + bf16-autocast + `torch.compile` recipe (measured 3.6x to 8.4x speedup, no metric drift). All default off, so existing pipelines stay bit-identical; compile is gated to inference/val/test and never fires during training.
- Added the streaming `AnomalyAUROCMetrics` node: pixel and image AUROC via torchmetrics `BinaryAUROC` accumulated across batches with O(thresholds) state, replacing a bespoke callback and the per-pixel CPU concat. A training-time monitoring metric; the authoritative AUROC remains the sklearn pass in the eval script.
- Retired the plugin-local selector: dropped `FixedHyperspectralSelector` and its test in favor of upstream `cuvis_ai.node.channel_selector.FixedWavelengthSelector` (n-channel, order-preserving, `normalize_output=False`), available in `cuvis-ai>=0.10.0`. The `examples` extra now floors `cuvis-ai>=0.10.0`.
- Restored legacy 3-channel input scaling: a uint8 RGB frame with max in `(1, 255]` divides by a fixed 255 again (a max-200 frame maps to 0.784, not 1.0); only reflectance input with max > 255 uses the per-cube max. Keeps existing 3-channel pipelines bit-identical.
- Single source for COCO helpers: `MultiFileNpzDataset` re-imports `_build_category_mask` / `_parse_coco_json` from `_coco_utils` instead of a diverged local copy, removing an undeclared `scikit-image` dependency (clean-install `ImportError`) and a test-vs-runtime builder mismatch.
- Migrated the README and example scripts from the removed `load_plugins` to `register_plugin` (cuvis-ai-core 0.10).
- Registered `AnomalyAUROCMetrics` in `examples/plugins.yaml`.
- Added bedding-anomaly train and inference tutorial notebooks under `notebooks/bedding_anomaly/`.
- Added a `no-local-sources` CI workflow that fails if `pyproject.toml` declares a local `[tool.uv.sources]` path entry (a machine-specific path must not ship in a release).

## 0.2.0 - 2026-06-23

- Migrated the example plugin manifest (`examples/plugins.yaml`) to the bare `capabilities:` shape required by cuvis-ai-schemas 0.6.0.
- Required `cuvis-ai-core>=0.10.0` and `cuvis-ai-schemas>=0.7.0`, adopting the released framework versions; `cuvis-ai-core>=0.10.0` carries the renamed `register_plugin(path)` plugin-registration API (was `register_plugins`), which the manifest-loading test now uses.
- Migrated the example scripts from the high-level `cuvis_ai.data.MultiFileCu3sDataModule` to `cuvis_ai_dataloader.data.MultiCu3sDataModule`, so the examples load cu3s data through the `cuvis-ai-dataloader` plugin instead of the high-level package. The `pin_memory` / `persistent_workers` / `worker_multiprocessing_context` loader options now go only to `MultiFileNpzDataModule` (which honors them); `MultiCu3sDataModule` rejects them as of dataloader 0.2.0.
- Declared `cuvis-ai-dataloader` as a provisioned plugin in `configs/plugins/cuvis_ai_dataloader.yaml` (repo + tag `v0.2.0`, `[cu3s, coco]` extras) instead of a package dependency, so this plugin's pyproject no longer hard-depends on a sibling plugin. The `examples` extra keeps only `cuvis-ai>=0.9.0` (which dropped the cuvis SDK), so neither the plugin nor its examples pull the cuvis SDK.

## 0.1.5 - 2026-06-10

- Required `cuvis-ai-core>=0.7.1` and `cuvis-ai-schemas>=0.5.2` (inherits the upstream security floors transitively).
- Updated `examples/plugins.yaml` `provides` entries to the `CatalogNodeEntry` `class_name:` form required by cuvis-ai-schemas 0.5.2.
- Added the `cuvis_ai_compat.yml` dependency-compatibility workflow (audits the plugin's deps against the cuvis-ai-core lock).
- Removed the PyPI/TestPyPI release workflow; the plugin is distributed via git tags referenced from cuvis-ai plugin manifests.
- Stripped `torch` / `torchvision` wheel hashes from `uv.lock`.

## 0.1.4 - 2026-05-11

- Switched the runtime dep from `opencv-python>=4.8.0` to `opencv-python-headless>=4.13.0.92` to match `cuvis-ai-sam3` / `cuvis-ai-adaclip`: the plugin has no `cv2.imshow` / window calls, so the GUI subdeps (`libGL`, `libGTK`) were dead weight.
- Fixed a silent `ImportError` in `MultiFileNpzDataModule` by inlining `_build_category_mask` and `_parse_coco_json` into `cuvis_ai_dinomaly/data/_coco_utils.py`; they were previously imported from `cuvis_ai.data.multi_file_dataset`, which does not exist in the released `cuvis-ai` package. Removed the `pytest.importorskip` guard that was hiding the failure in CI; the datamodule tests now run unconditionally.
- Added two unit tests for `_build_category_mask` (empty annotations -> zero mask; bbox annotation -> correct region fill).
- Removed the dead `cuvis-ai = { path = "../cuvis-ai", editable = true }` entry from `[tool.uv.sources]` in `pyproject.toml` (`cuvis-ai` was dropped as a runtime dep in 0.1.3 but its source override lingered).
- Added an inline comment in `pyproject.toml` explaining the `<3.12` Python cap: `anomalib==2.1.0` and `kornia==0.6.12` are tested on 3.11 only, and the kornia pin avoids a `kornia-rs` illegal instruction on CI runners.
- Added `pre_trained/` to `.gitignore`.
- Replaced hardcoded developer paths (`/home/dev/anish/...`, `/mnt/data/...`) in all six `configs/trainrun/*.yaml` files with placeholder comments.
- Rewrote `docs/publish_checklist.md` to be version-agnostic (`vX.Y.Z` placeholders, no personal paths) and added steps for the compatibility audit (section 8) and the registry update.
- Added coverage to the CI `test` job (`--cov=cuvis_ai_dinomaly --cov-report=xml --cov-report=term-missing --cov-fail-under=70`, coverage XML artifact uploaded for 7 days) and lowered `[tool.coverage.report] fail_under` from 90 to 70 to match the CI gate.
- Added `tags-ignore: ["v*.*.*"]` to `ci.yml` `on.push` so tag pushes no longer re-run CI (the new `release.yml` handles that).
- Added `.github/workflows/release.yml`: tag-triggered (`v*.*.*`), runs jobs `validate` -> `security` -> `build` (with a tag-vs-package-version check) -> `create-release` (extracts the matching CHANGELOG section as GitHub Release notes).
- Recorded a compatibility audit against `cuvis-ai-core` 0.1.0 and 0.5.2 in [`docs/compatibility_audit.md`](docs/compatibility_audit.md). Result: PASS: every shared dep (`numpy`, `tqdm`, `defusedxml`, `requests`) satisfies the plugin's specifier, and `anomalib`, `kornia`, `opencv-python-headless`, `open-clip-torch` are not in either core lock, so no conflict risk.
- Extended CI to adaclip-level: added `typecheck` (mypy, non-blocking) and `security` (pip-audit, detect-secrets, bandit) jobs, with the `build` job gating on all four hygiene jobs. Added `.secrets.baseline` (zero findings) and a `[tool.bandit]` config block; dev-deps grew with `mypy`, `pip-audit`, `detect-secrets`, and `bandit[toml]`.
- Added a `LICENSE` file (Apache-2.0 standard text plus Cubert GmbH copyright) at the repo root; `pyproject.toml` already declared `license = "Apache-2.0"` but the license text was not previously distributed.
- Added a "Plugin manifest" section to `README.md` documenting both the local-path and git-tag manifest forms (skill section 9 / "When to stop" requirement).
- Added a CI workflow (`ci.yml`) with `test`, `lint`, and `build` jobs (Ubuntu / Python 3.11 / `uv` with `--no-sources`); the test job runs `pytest tests/ -m "not slow"` so the `integration`-marked manifest-loading smoke test executes per the cuvis-ai plugin skill verification step.
- Dropped the runtime dependency on the high-level `cuvis-ai` package and added `cuvis-ai-schemas>=0.4.0`: the plugin now depends only on `cuvis-ai-core` plus `cuvis-ai-schemas`, matching `cuvis-ai-deepeiou` / `cuvis-ai-sam3`, and avoids transitively importing the proprietary Cuvis SDK (`cuvis_il`) at module load.
- Inlined a minimal `_LossNode` in `dinomaly_train_loss_bridge.py` mirroring `cuvis_ai.node.losses.LossNode` (a marker subclass that defaults `execution_stages = {TRAIN, VAL, TEST}` on `Node.__init__`). No behavior change for consumers.
- Added `requests>=2.31.0` to the runtime dependencies (an undeclared transitive of `anomalib==2.1.0` via the eager `anomalib.models.video.ai_vad.clip` import chain).
- Replaced the `tests/test_parity_markers.py` `importlib.spec_from_file_location` plus `exec_module` approach with `ast.parse`-based static inspection, so the marker checks no longer re-execute `test_parity.py` at collection time.
- Ran `ruff format` over `cuvis_ai_dinomaly/`, `tests/`, `examples/` and applied `ruff check --fix --unsafe-fixes` (4 import sorts plus 5 `dict()` -> `{}` rewrites in the example training scripts).

## 0.1.3 - 2026-05-04

- Added the `open-clip-torch>=2.24.0` runtime dependency: anomalib model modules import WinCLIP / OpenCLIP symbols during package initialization.

## 0.1.2 - 2026-05-04

- Pinned `kornia==0.6.12` to avoid the resolver selecting builds that pull `kornia-rs` and crash with an illegal instruction at runtime.

## 0.1.1 - 2026-05-04

- Pinned `anomalib==2.1.0` to avoid importing `kornia-rs`, which crashes with an illegal instruction in the plugin runtime.

## 0.1.0 - 2026-04-17

- Added the `cuvis_ai_dinomaly` plugin package with the `DinomalyDetector` and `DinomalyTrainLossBridge` node classes.
- Added plugin scaffolding with `pyproject.toml`, LICENSE attribution (Apache-2.0), and a README.
- Added an Anomalib `DinomalyModel` wrapper with a frozen DINOv2 encoder and a trainable bottleneck plus decoder.
- Added Hydra-based multifile training examples (RGB, CIR, concrete, joint, selector, frozen-AdaCLIP-bands).
- Added the `examples/plugins.yaml` local-path manifest exposing both nodes.
- Added a pytest suite with detector-forward, loss-bridge, port-contract, preprocess, manifest-load, and parity tests (slow tests gated behind `CUVIS_DINOMALY_SKIP_SLOW`).
