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

- **Inference** (`bedding_all6_inference_tutorial.ipynb`): `load_bedding_cube`
  downloads + crops a frame, builds the model batch directly from the cube (no
  NPZ), and runs the pretrained pipeline.
- **Training** (`bedding_all6_train_tutorial.ipynb`): `snapshot_download`s the
  dataset, then `convert_bedding_cu3s_to_npz.py` writes cropped per-frame NPZ +
  a splits CSV that `train_bedding_all6.py` consumes. NPZ conversion is currently
  required for training (no multi-file cu3s datamodule / center-crop node exists
  yet — see the "Future work" note in the notebook).

## Caveat: model artefacts are not on HF

The **dataset** is on HuggingFace, but the **pretrained pipeline**
(`dinomaly_bedding_all6.yaml` / `.pt`) and the eval-output artefacts
(`eval_val/report.json`, per-class JSON, ROC PNGs) live under
`/mnt/data/cuvis_ai_outputs/...` and are **not** distributed via HF. A fresh-clone
user must either run the training notebook to produce a pipeline, or obtain the
trained artefacts separately. Uploading the trained pipeline to an HF model repo
is future work.
