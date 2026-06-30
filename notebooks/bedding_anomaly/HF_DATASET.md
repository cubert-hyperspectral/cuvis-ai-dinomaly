# Bedding dataset on HuggingFace

The bedding dataset is published at
**[`cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding`](https://huggingface.co/datasets/cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding)**.
Both tutorial notebooks load from it by default — no local dataset copy required.

## Layout

```
data/train/<stem>.cu3s   (+ .json)      # 193 train frames (all-normal)
data/val/<stem>.cu3s     (+ .json)      #  59 val frames (normal + anomalous)
annotations_raw/labels/<stem>_mask.png  # GT masks (cube-side or RGB-side naming)
annotations_canonical/{train,val}_global_coco.json
splits.csv                              # split, stem, cu3s_path, coco_json_path,
                                        # image_id, filename_label, has_annotation,
                                        # category_ids, label_fault
class_map.json
```

- **Native resolution is 2400 × 4900.** The pretrained pipeline was trained on a
  center-crop to **1800 × 4300** (`cube[300:-300, 300:-300]`). `utils.load_bedding_cube`
  applies this crop automatically, and masks are cropped identically — so HF and
  the legacy local NPZ produce bit-identical model inputs.
- Masks use one of two naming patterns; `utils.load_bedding_mask_path` tries both
  (`<stem>_mask.png`, then `<stem>_<stem>_(0000|0)_RGB_mask.png`), matching
  `convert_bedding_cu3s_to_npz.py`'s `find_mask_png`.
- `splits.csv` flags the known annotation-gap frame (`frame_10`) with `label_fault=1`.

## Switching data source

Default is HuggingFace. To read a fast local mount on the dev server instead:

```bash
export BEDDING_DATA_SOURCE=local   # reads LOCAL_DATA_ROOT in utils.py
```

(or edit `BEDDING_DATA_SOURCE` / `LOCAL_DATA_ROOT` at the top of `utils.py`).
All four loaders honour the toggle: `load_bedding_cu3s_path`,
`load_bedding_mask_path`, `load_bedding_splits`, `load_bedding_cube`.

The cuvis SDK must be importable to read cu3s — set `CUVIS=/lib/cuvis` (or your
SDK dir) before launching Jupyter.

## How the notebooks use it

- **Inference** (`bedding_all6_inference_tutorial.ipynb`): `resolve_pipeline()`
  downloads the pretrained pipeline from the model repo (or uses your own via
  `BEDDING_PIPELINE_SOURCE=local`); `load_bedding_cube` downloads + crops a frame
  and builds the model batch directly from the cube (no NPZ). Self-contained.
- **Training** (`bedding_all6_train_tutorial.ipynb`): fully inline — downloads the
  dataset, converts cu3s → cropped NPZ, builds the 6-channel pipeline node-by-node,
  runs statistical-init + gradient training (`MAX_EPOCHS` knob), and saves the
  pipeline. No external scripts. Its output feeds straight back into the inference
  notebook via `BEDDING_PIPELINE_SOURCE=local`.

## Trained model on HuggingFace

The pretrained pipeline + validation metrics are published as a model repo:
**[`cubert-gmbh/dinomaly-bedding-all6`](https://huggingface.co/cubert-gmbh/dinomaly-bedding-all6)**
(`dinomaly_bedding_all6.yaml` + ~580 MB `.pt` + `eval_val/*.json`). The inference
notebook fetches it by default — no local artefacts required. To run your own
instead, train with the training notebook and set `BEDDING_PIPELINE_SOURCE=local`
(+ optionally `BEDDING_PIPELINE_DIR`).

Loading the pipeline requires the cuvis SDK + high-level `cuvis-ai` (it uses
`cuvis_ai.node.*` built-ins).
