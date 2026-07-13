# Lentils × Dinomaly — reproduction guide

How to reproduce the lentils foreign-object anomaly-detection experiments on the **current**
cuvis-ai stack, with three spectral front-ends (RGB / CIR / AdaCLIP-bands) plus an inference +
per-class-AUROC walkthrough.

> **Use `dinomaly`, not `dinomaly2`.** Everything here uses the current
> `cuvis_ai_dinomaly.DinomalyDetector`. An earlier `dinomaly2` run reported lentils pixel AUROC
> ~0.915 / image ~0.834, but that used the **defunct** `cuvis_ai_dinomaly2` package and its saved
> pipeline no longer loads — it is **not** a target for the current model. The data split and the
> baked masks are model-agnostic and remain valid; reproduce a number by **retraining**.

## Reproduce from scratch on a fresh machine (from HuggingFace)

End-to-end recipe, validated on a clean box (RTX 4090). All data comes from HuggingFace. The
commands are the same in bash and PowerShell; only the `CUVIS` env var is set differently (both
shown). Provisioning follows the standard Cuvis.AI install (canonical docs:
<https://docs.cuvis.ai/latest/>).

**0. Prerequisites**

- A CUDA GPU + recent driver.
- The **Cuvis C SDK** installed, with the `CUVIS` env var pointing at its dir (the `cuvis` Python
  bindings dynamically link `libcuvis`). The convert step reads cu3s, so this is required;
  training/inference on the NPZ is pure Python and does not need it.
- `uv`, `git`, an **HF token** (`huggingface-cli login`), and — only if you want the pipeline PNGs
  from the training scripts — `graphviz` (`sudo apt install graphviz`; otherwise the scripts just
  log a warning and skip the diagram).

**1. Clone + environment**

```bash
git clone <cuvis-ai-dinomaly>
git clone <cuvis-ai-dataloader>                       # alongside; [tool.uv.sources] points at it
cd cuvis-ai-dinomaly
uv sync --extra examples                              # framework + torch (cu128) + anomalib + cuvis-ai
uv pip install -e '../cuvis-ai-dataloader[cu3s,coco]' # cu3s reader + COCO labeler + cuvis bindings
huggingface-cli login                                 # HF token for the dataset download
# point CUVIS at the SDK dir (the dir containing libcuvis):
#   bash:        export CUVIS=/path/to/cuvis-sdk
#   PowerShell:  $env:CUVIS = "C:\path\to\cuvis-sdk"   (or persist: setx CUVIS "C:\path\to\cuvis-sdk")
```

(Once cuvis-ai-dataloader releases with the converter + `universe_csv`, swap the editable install
for `uv pip install 'cuvis-ai-dataloader[cu3s,coco]>=<version>'` and drop the local clone.)

**2. Download + convert from HF → per-frame NPZ**

The train-on-normals split is `splits_dinomaly.csv` on the HF dataset. Its train/val/test frames
(636) convert to per-frame NPZ (grouped by cu3s; each session downloaded once). Simplest driver:

```python
import utils  # notebooks/lentils_anomaly/utils.py
splits_json, universe_csv = utils.prepare_lentils_data("<npz_out_dir>")  # HF download + convert
```

`prepare_lentils_data` writes a **`universe.csv`** (`source, index, path`) + a baked
**`splits.json`** (see *Data model* below), reused on rerun. Tip: convert one session per
subprocess — the cuvis SDK aborts the *process* on session teardown *after* the files are written,
so per-session isolation keeps a long convert resumable + lossless.

**3a. Run the pedagogical notebooks (Jupyter)**

```bash
uv pip install jupyterlab       # notebook UI (kept out of the package deps)
uv run --no-sync jupyter lab    # then open notebooks/lentils_anomaly/lentils_rgb_train_tutorial.ipynb
```

- **`--no-sync` matters:** a plain `uv run` re-syncs from the lock and would uninstall the editable
  `[cu3s,coco]` extra and `jupyterlab` (neither is a package dependency). Running Jupyter from the
  project venv this way also means the default kernel already sees `cuvis_ai*` (no `ipykernel`
  registration).
- Every knob (`MAX_EPOCHS` / `IMAGE_SIZE` / `SMOKE_LIMIT`, and inference's `PIPELINE_DIR` /
  `TEST_LIMIT`) is a plain variable at the top of each notebook's setup cell; edit it in place.

**3b. Or run the scripts (full-fidelity 20/50-epoch runs)**

```bash
# 448px, 6 workers, AdamW, best-checkpoint. <UNIVERSE>/<SPLITS_JSON> = the prepare_lentils_data pair.
uv run python examples/train_dinomaly_rgb_multifile.py     data.universe_csv=<UNIVERSE> data.splits_json=<SPLITS_JSON> output_dir=<OUT>/rgb     training.max_epochs=20
uv run python examples/train_dinomaly_cir_multifile.py     data.universe_csv=<UNIVERSE> data.splits_json=<SPLITS_JSON> output_dir=<OUT>/cir     training.max_epochs=20
uv run python examples/train_dinomaly_rgb_frozen_adaclip_bands_multifile.py data.universe_csv=<UNIVERSE> data.splits_json=<SPLITS_JSON> output_dir=<OUT>/adaclip training.max_epochs=20
# inference — per-class AUROC on the 180-frame test (notebook, or the eval CLI):
uv run python examples/run_saved_dinomaly_pipeline_test_npz.py \
  --pipeline-yaml <OUT>/rgb/trained_models/dinomaly_multifile_rgb.yaml \
  --pipeline-pt   <OUT>/rgb/trained_models/dinomaly_multifile_rgb.pt \
  --universe-csv  <UNIVERSE> --splits-json <SPLITS_JSON> --output-dir <OUT>/rgb/eval
```

See *Scripts* below for the full-fidelity knobs and the learnable-Concrete variant.

## Stack

| package | version |
|---|---|
| cuvis-ai | ≥ 0.10 |
| cuvis-ai-core | ≥ 0.11 |
| cuvis-ai-schemas | ≥ 0.8 |
| cuvis-ai-dataloader | with the `universe_csv` split model (provides `MultiNpzDataModule` / `npz_multi`) |
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

## Data model: `universe.csv` + `splits.json` (the selector split model)

`prepare_lentils_data(npz_dir)` (in `utils.py`) downloads the cu3s dataset from HuggingFace (via
`PublicDatasets`), converts the `splits_dinomaly.csv` frames to per-frame NPZ with
cuvis-ai-dataloader's `convert_split_manifest`, and returns `(splits_json, universe_csv)`:

- **`universe.csv`** (`source, index, path`) is the sample universe: one row per frame, `source` the
  posix cu3s path, `index` the measurement index (`local_image_id`, also the COCO `image_id`), `path`
  the per-frame `.npz` relative to the CSV. Each NPZ carries `cube [H,W,61]`, `wavelengths [61]`, and
  — for annotated frames — a baked binary `mask [H,W]` + category `class_mask [H,W]` (normal frames
  have no mask key; the loader emits zeros).
- **`splits.json`** is the baked assignment: a core `DataSplitConfig` of `file_indices` selectors
  (one per source per split). It is identical to the dataset's shipped `splits/dinomaly.json`, so the
  same file drives training here and in the CuvisNEXT split designer. `MultiNpzDataModule` resolves
  it over the `universe.csv`.

Verified end-to-end **on a fresh machine at 20 epochs** for all three variants (see *Validated
results* below): normal frames → empty mask, annotated frames → baked class ids matching the GT
`category_labels`; validated 180/180 byte-for-byte vs the GT COCO. The convert step needs the cu3s
reader stack (`cuvis` SDK + the `[cu3s,coco]` extra); training/inference on the NPZ needs neither.

## Notebooks (pedagogical)

In `notebooks/lentils_anomaly/`. Run from that folder (they import the sibling `utils.py` and the
repo's `examples/plugins.yaml`); see *Reproduce from scratch → 3a* for the Jupyter launch. Every
knob is a plain variable at the top of each notebook's setup cell (no env vars).

- `lentils_rgb_train_tutorial.ipynb` — fixed-wavelength RGB selector (650 / 550 / 450 nm).
- `lentils_cir_train_tutorial.ipynb` — CIR selector (NIR 860 / Red 670 / Green 560 nm).
- `lentils_adaclip_bands_train_tutorial.ipynb` — the 3 bands AdaCLIP's frozen concrete selector
  converged to (cube indices 14/59/57), resolved to wavelengths from the data (≈542/902/886 nm) and
  used as a **fixed** `FixedWavelengthSelector`. Head-to-head-with-AdaCLIP variant (same bands, but
  reconstructed by Dinomaly). A fully *learnable* concrete selector is the standalone script below.
- `lentils_inference_tutorial.ipynb` — load a trained pipeline, evaluate on the 180-frame test,
  report overall pixel + image AUROC and a **per-class pixel AUROC** breakdown (from `class_mask`).

Each train notebook saves `<OUTPUT_DIR>/trained_models/<name>.yaml` + `.pt`; point the inference
notebook at it via its `PIPELINE_DIR` variable.

## Scripts (full-fidelity runs)

The Hydra scripts in `examples/` are the canonical path for the definitive 50-epoch numbers
(they add the LR scheduler + all knobs the notebooks omit for clarity). The NPZ backend takes a
`data.universe_csv` + `data.splits_json` pair (a `convert_split_manifest` output); the cu3s backend
takes `data.splits_csv` instead.

```bash
# RGB (fixed wavelengths). <UNIVERSE>/<SPLITS_JSON> = a convert_split_manifest pair; <OUT> = output dir.
uv run python examples/train_dinomaly_rgb_multifile.py \
  output_dir=<OUT>/dinomaly_rgb_50ep \
  training.max_epochs=50 \
  data.universe_csv=<UNIVERSE> data.splits_json=<SPLITS_JSON> \
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

## Validated results (fresh machine, 20 epochs, from HF, 180-frame test)

| variant | pixel AUROC | image AUROC |
|---|---|---|
| RGB | 0.944 | 0.728 |
| CIR | 0.994 | 0.893 |
| AdaCLIP-bands | 0.994 | 0.929 |

Per-class pixel AUROC is 0.90–1.00 across the categories (`blue_paper` ≈ 1.0 for all variants). CIR
and AdaCLIP-bands clearly beat plain RGB; AdaCLIP-bands has the best image-level AUROC. These are
20-epoch numbers — a full 50-epoch run would likely improve them. (RGB pixel 0.944 already
matches/exceeds the old defunct-`dinomaly2` 0.915, which is *not* a target — see the
dinomaly-not-dinomaly2 note above.)

## What is verified

- The full path — **HuggingFace → convert → train → inference** — runs end-to-end on a *fresh
  machine* (fresh clone, freshly-provisioned SDK) for all three variants, with no code changes.
- Baked masks match the GT `category_labels` across day2/day3/day4 sessions + a no-COCO control.
- All four notebooks run end-to-end; the scripts run the full 20-epoch training + eval.
