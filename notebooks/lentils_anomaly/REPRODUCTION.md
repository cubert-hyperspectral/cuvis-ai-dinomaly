# Lentils × Dinomaly — reproduction guide

How to reproduce the lentils foreign-object anomaly-detection experiments on the **current**
cuvis-ai stack, with three spectral front-ends (RGB / CIR / AdaCLIP-bands) plus an inference +
per-class-AUROC walkthrough.

> **Use `dinomaly`, not `dinomaly2`.** Everything here uses the current
> `cuvis_ai_dinomaly.DinomalyDetector`. An earlier `dinomaly2` run reported lentils pixel AUROC
> ~0.915 / image ~0.834, but that used the **defunct** `cuvis_ai_dinomaly2` package and its saved
> pipeline no longer loads — it is **not** a target for the current model. The data split and the
> baked masks are model-agnostic and remain valid; reproduce a number by **retraining**.

## Stack

| package | version |
|---|---|
| cuvis-ai | ≥ 0.10 |
| cuvis-ai-core | ≥ 0.10 |
| cuvis-ai-schemas | ≥ 0.7 |
| cuvis-ai-dataloader | ≥ 0.3 (provides `MultiNpzDataModule` / `npz_multi`) |
| anomalib | 2.1 |

A CUDA GPU is required (DINOv2 reg ViT-B/14 encoder, ~148 M params, ~592 MB pipeline).

## Dataset + split

61-channel VNIR (430–910 nm) cubes; foreign objects (8 COCO categories, 0 = normal) are the
anomalies. Published: `cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils`.

Dinomaly is unsupervised and reconstruction-based, so it **trains on normal frames only**. The
train-on-normals split (`splits_dinomaly.csv` on HF):

| split | frames | anomalous | role |
|---|---|---|---|
| `train` | 308 | 0 | Dinomaly training |
| `val` | 148 | 84 | calibration / threshold |
| `test` | 180 | 112 | evaluation |
| `adaclip_train` | 500 | 500 | supervised baseline's positives — **held out** from Dinomaly |

`val` (148) and `test` (180) are identical to the supervised **adaclip** baseline's split, so the
two models stay directly comparable.

## Data: local (verified) vs HuggingFace (pending)

The datamodule reads a splits CSV of `(split, npz_path, image_id)`; each NPZ carries
`cube [H,W,61]`, `wavelengths [61]`, and — for annotated frames — a baked binary `mask [H,W]` and
category `class_mask [H,W]` (normal frames have no mask key; the loader emits zeros).

- **`local` (default, verified).** The on-server per-frame NPZ already carry correct baked masks
  (validated 180/180 byte-for-byte against the GT COCO). The train/inference notebooks read them
  via the local split CSV (`diagnostics/lentils_splits_npz_dinomaly.csv`). Set
  `LENTILS_DATA_SOURCE=local` (the default).
- **`hf` (download → convert) — verified.** `ensure_lentils_npz()` downloads the cu3s + their
  per-session COCO (`json_path`) from HF and converts to NPZ. Each per-session cu3s is indexed by
  its **measurement index = `local_image_id`**, which is also the `image_id` in that session's
  COCO — so frames are read + labelled by `local_image_id` (`camera_frame_num` is the original
  camera counter, *not* the cu3s index). Verified end-to-end on a subset: normal frames → empty
  mask, annotated frames → baked class ids matching the GT `category_labels`.
  - **Extra deps:** this path reads cu3s, so it needs the cu3s reader stack — `cuvis` SDK +
    `dataclass_wizard` (`uv pip install 'cuvis-ai-dataloader[cu3s]'`). The `local` path doesn't.
  - **Note:** the cuvis SDK may abort the *process* on session teardown (after all files are
    written) — harmless for a convert-then-train workflow (convert is its own step), but keep the
    convert separate from the training process.

## Notebooks (pedagogical)

In `notebooks/lentils_anomaly/`. Run from that folder (they import the sibling `utils.py` and the
repo's `examples/plugins.yaml`). All knobs are env-overridable:

| env var | default | meaning |
|---|---|---|
| `LENTILS_DATA_SOURCE` | `local` | `local` NPZ, or `hf` (download→convert; needs `cuvis-ai-dataloader[cu3s]`) |
| `LENTILS_MAX_EPOCHS` | `1` | bump to `50` for a full run |
| `LENTILS_IMAGE_SIZE` | `448` | square side, multiple of 14 |
| `LENTILS_SMOKE_LIMIT` | `0` | `0` = all frames; `N` = N per split (fast dry-run) |
| `LENTILS_PIPELINE_DIR` | RGB run's `trained_models` | which trained pipeline inference loads |
| `LENTILS_TEST_LIMIT` | `0` | inference: `0` = full 180, `N` = first N frames |

- `lentils_rgb_train_tutorial.ipynb` — fixed-wavelength RGB selector (650 / 550 / 450 nm).
- `lentils_cir_train_tutorial.ipynb` — CIR selector (NIR 860 / Red 670 / Green 560 nm).
- `lentils_adaclip_bands_train_tutorial.ipynb` — the 3 bands AdaCLIP's frozen concrete selector
  converged to (cube indices 14/59/57), resolved to wavelengths from the data (≈542/902/886 nm) and
  used as a **fixed** `FixedWavelengthSelector`. Head-to-head-with-AdaCLIP variant (same bands, but
  reconstructed by Dinomaly). A fully *learnable* concrete selector is the standalone script below.
- `lentils_inference_tutorial.ipynb` — load a trained pipeline, evaluate on the 180-frame test,
  report overall pixel + image AUROC and a **per-class pixel AUROC** breakdown (from `class_mask`).

Each train notebook saves `<OUTPUT_DIR>/trained_models/<name>.yaml` + `.pt`; point the inference
notebook at it with `LENTILS_PIPELINE_DIR`.

## Scripts (full-fidelity runs)

The Hydra scripts in `examples/` are the canonical path for the definitive 50-epoch numbers
(they add the LR scheduler + all knobs the notebooks omit for clarity). The split CSV's backend is
auto-detected: a `npz_path` column → `MultiNpzDataModule`, else `MultiCu3sDataModule`.

```bash
# RGB (fixed wavelengths). <SPLITS> = your (split, npz_path, image_id) CSV; <OUT> = output dir.
uv run python examples/train_dinomaly_rgb_multifile.py \
  output_dir=<OUT>/dinomaly_rgb_50ep \
  training.trainer.max_epochs=50 \
  data.splits_csv=<SPLITS> \
  data.num_workers=6 data.persistent_workers=true eval_mode=best

# CIR (NIR / Red / Green) — same, via:
uv run python examples/train_dinomaly_cir_multifile.py  <same overrides>

# AdaCLIP frozen bands (fixed selector on indices 14/59/57) — STANDALONE script:
uv run python examples/train_dinomaly_rgb_frozen_adaclip_bands_multifile.py <same overrides>

# Concrete (LEARNABLE band selector, joint + distinctness) — STANDALONE script:
uv run python examples/train_dinomaly_concrete_joint_multifile.py <same overrides>
```

> The shared `dinomaly_multifile_train_common.py` wires only the fixed **RGB / CIR** selectors and
> now **raises** on any other `band_mode`. The Concrete selector needs a different graph
> (`selection_weights → distinctness loss`, two loss nodes), so it lives entirely in the
> standalone `train_dinomaly_concrete_joint_multifile.py`. The older
> `train_dinomaly_concrete_multifile.py` / `_concrete_selector_multifile.py` route through the
> shared trainer and therefore now fail fast — prefer the joint script.

### Evaluate a saved pipeline (CLI, mirrors the inference notebook)

```bash
uv run python examples/run_saved_dinomaly_pipeline_test_npz.py \
  --pipeline-yaml <run>/trained_models/<name>.yaml \
  --pipeline-pt   <run>/trained_models/<name>.pt \
  --splits-csv    <SPLITS> \
  --output-dir    <run>/eval_test
```

## What is verified here

- All four notebooks run end-to-end on the current stack (1-epoch × few-frame smokes): each train
  notebook builds its pipeline, trains, and saves a loadable `.yaml`/`.pt`; the inference notebook
  loads a saved pipeline and computes overall + per-class AUROC on a mixed test subset.
- Smoke numbers are meaningless by construction (1 epoch, tiny subset). For a real, comparable
  result run the full 50 epochs on the complete split.
