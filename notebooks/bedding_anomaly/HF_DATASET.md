# Bedding dataset on HuggingFace

The bedding dataset is published at
**[`cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding`](https://huggingface.co/datasets/cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding)**.
Both tutorial notebooks download it inline with `PublicDatasets.download_dataset` (registered as
`industrial_fod_bedding`); no local dataset copy or environment variable is required. The full
snapshot is ~170 GB, so plan disk space accordingly.

## Layout

```
data/train/<stem>.cu3s   (+ <stem>.json)  # 193 train frames (all-normal)
data/val/<stem>.cu3s     (+ <stem>.json)  #  59 val frames (normal + anomalous)
annotations_raw/labels/<stem>_mask.png    # GT masks (cube-side or RGB-side naming)
annotations_canonical/{train,val}_global_coco.json
splits.csv                                # split, stem, cu3s_path, coco_json_path,
                                          # image_id, filename_label, has_annotation,
                                          # category_ids, label_fault
class_map.json
```

- `data/{split}/<stem>.json` are **per-session COCO annotations**, one sibling json per cu3s.
  The NPZ converter consumes them directly: each cu3s holds a single measurement, and the
  sibling COCO's single image record has `id = 0`, matching the measurement index.
- `splits.csv` predates the converter's manifest column contract (`cu3s_path, local_image_id,
  split` plus optional `json_path`), so the notebooks rename `image_id` to `local_image_id` and
  `coco_json_path` to `json_path` inline (a two-column pandas rename) before handing it to
  `convert_split_manifest`.
- **Native resolution is 2400 x 4900.** The pretrained pipeline was trained on a center-crop to
  **1800 x 4300** (`crop=(300, 300, 300, 300)` margins). The notebooks pass that crop to the
  converter, and masks are cropped identically, so converted NPZ match the published model's
  training inputs.
- The cu3s files store the already-processed cube, so the notebooks convert with
  `processing_mode=None` (read as-is); reprocessing would break parity with the published model.
- Masks use one of two naming patterns (`<stem>_mask.png`, or
  `<stem>_<stem>_(0000|0)_RGB_mask.png`); the converter reads the per-session COCO jsons, so the
  raw PNGs are reference material only.
- `splits.csv` flags the known annotation-gap frame (`frame_10`) with `label_fault=1`.

## How the notebooks use it

Both notebooks provision data with the same inline pattern:
`PublicDatasets.download_dataset("industrial_fod_bedding", ...)` fetches the raw snapshot, then
`convert_split_manifest` (from cuvis-ai-dataloader) converts the renamed `splits.csv` manifest to
per-frame NPZ, emitting a `universe.csv` + `splits.json` for `MultiNpzDataModule`. Conversion is
skipped when the artifacts already exist with the expected frame counts.

- **Training** (`bedding_all6_train_tutorial.ipynb`): converts **train + val** (193 + 59 frames)
  to `outputs/npz_local`, builds the 6-channel pipeline node-by-node, runs statistical-init +
  gradient training (`MAX_EPOCHS` knob), and saves the pipeline under
  `outputs/trained_run/trained_models`.
- **Inference** (`bedding_all6_inference_tutorial.ipynb`): converts **val only** (59 frames) to
  `outputs/npz_val` and fetches the published pipeline (yaml + weights) via `hf_hub_download`
  from the model repo below. To evaluate your own training run instead, point the notebook's
  `PIPE_YAML` / `PIPE_PT` at the train notebook's `outputs/trained_run/trained_models` (a comment
  in the configuration cell marks the spot).

## Trained model on HuggingFace

The pretrained pipeline + validation metrics are published as a model repo:
**[`cubert-gmbh/dinomaly-bedding-all6`](https://huggingface.co/cubert-gmbh/dinomaly-bedding-all6)**
(`dinomaly_bedding_all6.yaml` + ~580 MB `dinomaly_bedding_all6.pt` + `eval_val/*.json`). The
inference notebook fetches the yaml + pt by default; no local artefacts required.

Loading the pipeline requires the cuvis SDK + high-level `cuvis-ai` (it uses
`cuvis_ai.node.*` built-ins).
