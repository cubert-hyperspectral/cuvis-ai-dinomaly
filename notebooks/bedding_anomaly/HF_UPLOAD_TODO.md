# Bedding dataset — HuggingFace upload TODO

The bedding × Dinomaly tutorial notebooks (`utils.load_bedding_cu3s_path`) are
designed to switch seamlessly from local-disk to HuggingFace once the bedding
dataset is uploaded. This doc captures the expected repo structure so whoever
performs the upload can match it.

## Target HF repo

- **Repo id:** `cubert-hyperspectral/bedding-6ch`
- **Type:** dataset
- **License:** to be decided (likely CC-BY-4.0 for the cu3s session files plus
  the labels, matching the lentils dataset).

## Expected repo layout

```
cubert-hyperspectral/bedding-6ch/
├── README.md                                     # dataset card
├── splits.csv                                    # cube paths + class labels + split assignment
├── exported/
│   ├── train/   <ts>_<frame>_<label>.cu3s        # 193 files
│   └── val/     <ts>_<frame>_<label>.cu3s        # 59 files
├── labels/
│   └── <ts>_<frame>_<label>_mask.png             # binary mask, same H×W as cube
└── labels_extracted/
    └── <ts>_<frame>_<label>.json                 # COCO-style annotation source
```

Filename pattern: `<YYYYMMDD>_<HHMMSS>_frame_<N>_<labelpair>_<recipeA>_<recipeB>.cu3s`
where `<labelpair>` is one of `nok_nok`, `ok_nok`, `nok_ok`, `ok_ok`. Frames
containing `_ok_ok_` are normal; everything else is anomalous (this is EAD's
labelling convention).

## What the notebook loader expects

`utils.load_bedding_cu3s_path(frame_stem)` resolves a stem like
`20250310_151720_frame_104_ok_nok_rdx_rwx` to a cu3s file on disk. The
expected HF path is `exported/val/<stem>.cu3s` (or `exported/train/...` —
loader auto-detects based on the splits CSV).

When the upload happens, edit `utils.py`:

```python
# 1. set BEDDING_HF_REPO_ID (already set to the expected id)
# 2. flip BEDDING_HF_FALLBACK to default-True OR drop the env-guard
# 3. inside load_bedding_cu3s_path, replace the NotImplementedError
#    branch with:
return Path(hf_hub_download(
    repo_id=BEDDING_HF_REPO_ID,
    repo_type="dataset",
    filename=f"exported/val/{frame_stem}.cu3s",
    cache_dir=str(BEDDING_HF_CACHE),
))
```

That's the entire diff to make all three tutorial notebooks remote-ready.

## What else may need to move

- **Mask PNGs** — currently at `/mnt/data/bedding_dataset/labels_extracted/labels/`.
  After HF upload, the inference + results notebooks would call
  `hf_hub_download(... filename=f"labels/{stem}_mask.png")` instead. Add a
  matching `load_bedding_mask_png` helper to `utils.py` at upload time.
- **Saved pipeline** — `dinomaly_bedding_all6.{yaml,pt}` for the high-res 20-ep
  model. Either keep it under a Hub model repo (e.g. `cubert-hyperspectral/dinomaly-bedding-all6-highres`)
  alongside the dataset, or ship as a release artefact on the cuvis-ai-dinomaly
  GitHub repo. Either way, notebooks should call a single `load_pipeline()`
  helper that abstracts the source.
- **frame_10 annotation gap** — `20250311_101035_frame_10_nok_ok_rdx_rwx.json`
  has zero polygons, so its mask PNG was never generated. Fix this before
  upload, otherwise downstream metric comparisons inherit the same
  mask-vs-filename-label discrepancy we tracked in
  `comparisons/ead_baseline/discrepancy_investigation.md`.

## Reference

- Lentils dataset (the analog already on HF, used by `notebooks/lentils_sliding/`)
  for the structural template.
- `comparisons/headline_matrix.md` for the final metric table the dataset card
  should cite when describing the bedding × Dinomaly benchmarks.
