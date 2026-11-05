# Lentils × Dinomaly — training scripts (RGB / CIR / AdaCLIP-bands)

How to run and understand the three lentils foreign-object anomaly-detection training scripts on
the **current** cuvis-ai stack. All three train the same model (**Dinomaly**) on the same data;
they differ only in the **spectral front-end** that reduces the 61-band cube to the 3 channels the
DINOv2 encoder expects.

> **Use `dinomaly`, not `dinomaly2`.** Everything here uses `cuvis_ai_dinomaly.DinomalyDetector`.
> An earlier `dinomaly2` run reported lentils pixel AUROC ~0.915, but that used the **defunct**
> `cuvis_ai_dinomaly2` package and its saved pipeline no longer loads — it is **not** a target for
> the current model. The data split + masks are model-agnostic; reproduce a number by retraining.

---

## 1. The three scripts

| variant | script (`examples/`) | Hydra config (`configs/trainrun/`) | selector |
|---|---|---|---|
| **RGB** | `train_dinomaly_rgb_multifile.py` | `dinomaly_multifile_rgb.yaml` | `FixedWavelengthSelector(650, 550, 450 nm)` |
| **CIR** | `train_dinomaly_cir_multifile.py` | `dinomaly_multifile_cir.yaml` | `CIRSelector(NIR 860 / Red 670 / Green 560 nm)` |
| **AdaCLIP-bands** | `train_dinomaly_rgb_frozen_adaclip_bands_multifile.py` | `dinomaly_multifile_rgb_frozen_adaclip_bands.yaml` | `FixedWavelengthSelector` on AdaCLIP's frozen bands (cube indices **14/59/57** → ~**542/902/886 nm**) |

- **RGB** and **CIR** share the trainer module `dinomaly_multifile_train_common.py`
  (`run_dinomaly_multifile_training(cfg, band_mode="rgb"|"cir", ...)`).
- **AdaCLIP-bands** is **standalone** (it resolves the band indices to wavelengths from the first
  train frame at runtime, then builds a fixed selector on them).

There are also `train_dinomaly_concrete_*_multifile.py` scripts (a *learnable* Concrete band
selector). Only `train_dinomaly_concrete_joint_multifile.py` is wired up; the other two route
through the RGB/CIR common and now raise. They are not part of this trio.

---

## 2. What Dinomaly does here (understanding)

Dinomaly is an **unsupervised, reconstruction-based** anomaly detector: a frozen **DINOv2 reg
ViT-B/14** encoder + a trainable bottleneck & decoder that learn to reconstruct **normal** feature
maps. At test time, poorly-reconstructed regions score high → anomaly map. Because it models
*normality*, it **trains on normal frames only**; foreign objects appear only at val/test.

The pipeline graph (identical across the three, only the selector node changes):

```
LentilsAnomalyDataNode(normal_class_ids=[0])
    → MinMaxNormalizer (running stats; statistically initialised before gradient training)
    → <selector>  (RGB / CIR / AdaCLIP-bands → 3 channels)
    → DinomalyDetector (DINOv2 encoder frozen; bottleneck + decoder trained)
    → QuantileBinaryDecider(0.995)      → AnomalyDetectionMetrics (IoU/Dice/… drives checkpointing)
                                        → AnomalyAUROCMetrics (per-epoch pixel/image AUROC)
    → TensorBoardMonitorNode
DinomalyDetector.training_loss → DinomalyTrainLossBridge (the training loss node)
```

Training is two-phase: `StatisticalTrainer` initialises the MinMax normaliser's bounds
(`minmax_init_frames` frames), then `GradientTrainer` trains the unfrozen nodes
(`unfreeze_nodes: [dinomaly_detector]`). Best checkpoint is selected on `metrics_anomaly/iou`.

---

## 3. Environment

| package | version |
|---|---|
| cuvis-ai | ≥ 0.16.1 |
| cuvis-ai-core | ≥ 0.19.3 (the post-fit calibration reuses the fit's open recordings; an 8 GB GPU with the SDK on the GPU needs it) |
| cuvis-ai-schemas | ≥ 0.12.0 |
| cuvis-ai-dataloader | ≥ 0.8.2 (`MultiCu3sDataModule` with a bounded set of open recordings, `read_ahead`, `cuda_cubes`) |
| anomalib | 2.1 |

- A **CUDA GPU** is required (~148 M params, ~592 MB saved pipeline).
- The **Cuvis C SDK** reads the cu3s recordings: set `CUVIS=/lib/cuvis` (path to the SDK) in the
  environment.
- The Dinomaly plugin is registered from `configs/plugins/dinomaly.yaml` (the scripts do this themselves).

---

## 4. Data

### 4a. What you pass in `data`

The scripts read the cu3s recordings directly with cuvis-ai-dataloader's `MultiCu3sDataModule`
(`data_module: cu3s_multi`, `processing_mode: Reflectance`). Nothing is converted beforehand. The
HF dataset ships both inputs next to its `data/` folder:

- `data.universe_csv`: `universe.csv`, one row per frame (`source, index, annotation, group,
  tags`: the recording, the frame's index in it, its COCO file).
- `data.splits_json`: `splits/dinomaly.json`, a core `DataSplitConfig` whose `file_indices`
  selectors assign train/val/test. Without it, the universe.csv needs a `split` column.
- `data.reader`: the reader options, passed to `MultiCu3sDataModule` as they are (see 4b).

### 4b. Reader settings

`data.num_workers` stays 0: the SDK reads in the training process, and DataLoader workers cannot
use it on the GPU (on Windows they fail outright, on Linux CUDA does not initialise in a forked
worker). The `data.reader` defaults fit an 8 GB GPU (a laptop):

| key | default | meaning |
|---|---|---|
| `max_open_sessions` | 3 | recordings kept open per loader; every open one holds about 0.3 to 0.6 GB of GPU memory, and the train and the val loader each keep their own set. More open recordings mean fewer reopens (about 1 to 2 s each) during a shuffled epoch |
| `read_ahead` | 1 | frames read ahead of the model step, on reader threads |
| `cuda_cubes` | false | `true` keeps the cubes on the GPU instead of a round trip through host memory |
| `sdk_cuda` | true (dataloader default) | `false` processes the cubes on the CPU: no extra GPU memory, but slower |

On a 24 GB GPU add `data.reader.max_open_sessions=6`: in a 5-epoch RGB run on an RTX 4090 an
epoch took 89 s instead of 176 s, at 13 GB of GPU memory instead of 9 GB. Keeping all 15
recordings open is faster still but needs about 21 GB, and with `read_ahead: 2` and
`cuda_cubes: true` together it crashed the Cuvis SDK in longer runs (an illegal memory access in
its GPU memory pool, after 1 to 7 epochs).

Each batch carries `cube` `[B, H, W, 61]` float32, `wavelengths` `[B, 61]` int32 and `mask`
`[B, H, W]` int32 with the per-pixel COCO category ids (0 = background; a recording without a
labels file gives zeros). `LentilsAnomalyDataNode(normal_class_ids=[0])` turns `mask` into the
binary anomaly mask. A frame's COCO `image_id` is its read `index`.

### 4c. The Dinomaly split (train-on-normals)

Published as `splits/dinomaly.json` (a selector over the shipped `universe.csv`) on the HF dataset
`cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils`:

| split | frames | anomalous | role |
|---|---|---|---|
| `train` | 308 | 0 | Dinomaly training |
| `val` | 148 | 84 | calibration / threshold |
| `test` | 180 | 112 | evaluation |
| `adaclip_train` | 500 | 500 | supervised AdaCLIP baseline's positives — **held out** from Dinomaly |

`val` (148) and `test` (180) are identical to the supervised AdaCLIP baseline's split, so the two
models are directly comparable.

---

## 5. How to run

From the repo root. `<LENTILS>` = the HF lentils download
(`XMR_Industrial_Foreign_Object_Detection_Lentils/`), `<INDEX>` = `<LENTILS>/universe.csv`,
`<SPLITS_JSON>` = `<LENTILS>/splits/dinomaly.json`, `<OUT>` = output dir.

```bash
# RGB (fixed 650/550/450)
uv run python examples/train_dinomaly_rgb_multifile.py \
  data.universe_csv=<INDEX> data.splits_json=<SPLITS_JSON> output_dir=<OUT>/dinomaly_rgb \
  training.max_epochs=50 eval_mode=best

# CIR (NIR/Red/Green)
uv run python examples/train_dinomaly_cir_multifile.py \
  data.universe_csv=<INDEX> data.splits_json=<SPLITS_JSON> output_dir=<OUT>/dinomaly_cir \
  training.max_epochs=50 eval_mode=best

# AdaCLIP frozen bands (indices 14/59/57 → nm resolved from the data)
uv run python examples/train_dinomaly_rgb_frozen_adaclip_bands_multifile.py \
  data.universe_csv=<INDEX> data.splits_json=<SPLITS_JSON> output_dir=<OUT>/dinomaly_adaclip \
  training.max_epochs=50
```

Any config key is overridable on the command line (Hydra). Useful ones:
`data.reader.max_open_sessions`, `data.reader.read_ahead`, `data.reader.cuda_cubes` (section 4b),
`dinomaly.image_size` (448 default, multiple of 14), `minmax_init_frames`. Note: `eval_mode` applies to the RGB/CIR scripts (the AdaCLIP-bands script
always evaluates with the best checkpoint).

---

## 6. Key hyperparameters (from the configs)

| | value |
|---|---|
| encoder | `dinov2reg_vit_base_14` (frozen) |
| bottleneck dropout / decoder depth | 0.2 / 8 |
| image_size / crop_size | 448 / 448 (`use_center_crop: false`) |
| optimizer | AdamW, lr 2e-3, weight_decay 1e-4, betas (0.9, 0.999) |
| scheduler | ReduceLROnPlateau on `metrics_anomaly/iou` (mode max) |
| gradient_clip_val | 0.1 |
| max_epochs | 50 |
| batch_size | 1 |
| minmax_init_frames | 20 |
| seed | 42 |
| checkpoint | monitor `metrics_anomaly/iou` (max), save_top_k=1, save_last |

---

## 7. Outputs + evaluation

Each run writes to `<OUT>/...`:

- `trained_models/<pipeline_name>.yaml` + `.pt` — the deployable pipeline (load with
  `CuvisPipeline.load_pipeline`).
- `checkpoints/` — Lightning checkpoints (best + last; large — safe to delete after export).
- `tensorboard/` — training curves (`uv run tensorboard --logdir=<OUT>/tensorboard`).
- `pipeline/<name>.png` — rendered pipeline graph.
- Val/test metrics are logged at the end of the run (in stdout / the run log).

**Evaluate a saved pipeline on the 180-frame test** (overall + per-frame):

```bash
uv run python examples/run_saved_dinomaly_pipeline_test_npz.py \
  --pipeline-yaml <OUT>/trained_models/<name>.yaml \
  --pipeline-pt   <OUT>/trained_models/<name>.pt \
  --universe-csv     <INDEX> --splits-json <SPLITS_JSON> \
  --output-dir    <OUT>/eval_test
```

For a **per-class pixel AUROC** breakdown (from the per-pixel COCO category ids), use the
`notebooks/lentils_anomaly/lentils_inference_tutorial.ipynb` notebook (set
`PIPELINE_DIR=<OUT>/trained_models`).

---

## 8. Notes / gotchas

- **GPU memory.** The open recordings live next to the model on the GPU, in the train and in
  the val loader. Lower `data.reader.max_open_sessions` first when memory is short; at batch 1
  the model step, not the read, sets the pace once `read_ahead` is on.
- **Single GPU** → run one training at a time.
- **Comparability.** RGB / CIR / AdaCLIP-bands train on the identical split, so their test metrics
  are directly comparable to each other and (on val/test) to the supervised AdaCLIP baseline.
- **AdaCLIP-bands vs learnable concrete.** This trio uses the AdaCLIP-selected bands as a *fixed*
  selector. To *learn* the 3 bands jointly instead, see the standalone
  `train_dinomaly_concrete_joint_multifile.py`.
