# EfficientAD baseline on bedding val

Reproduces the **0.91 pixel-AUROC** baseline using the supplied `EAD_model.ckpt`.

## What's in here

| file | source |
|---|---|
| `reporting.py` | vendored from `cuvis.ai.examples/EfficientAD/reporting.py` with bedding-compatibility patches (see `PATCHES.md`) |
| `example_report_config.yaml` | matching config, points at `/mnt/data/bedding_dataset/exported/val` |
| `annotations.json` | 24-class label map (background + 23 anomaly classes) |

## Prerequisites

1. Bedding val cubes extracted at `/mnt/data/bedding_dataset/exported/val/` (59 `.cu3s` files).
2. PNG masks at `/mnt/data/bedding_dataset/labels_extracted/labels/`.
3. Checkpoint at `/mnt/data/bedding_dataset/EAD_model.ckpt` (md5 `b74e44b93948bfe89f0988512cfa359d`).
4. Clone the upstream `cuvis.ai.examples` repo for the `EfficientAD/` module imports
   (`EfficientAD_lightning`, `EfficientADCuvisDataSet`); this `reporting.py` must
   run from inside that repo's `EfficientAD/` folder.

## Run

```bash
# 1. Make mask symlinks so EAD's dataset class finds the PNGs (51 links)
bash ../ead_make_mask_symlinks.sh \
    /mnt/data/bedding_dataset/exported/val \
    /mnt/data/bedding_dataset/labels_extracted/labels

# 2. Drop the patched files into the upstream repo
cp reporting.py example_report_config.yaml annotations.json \
   /path/to/cuvis.ai.examples/EfficientAD/

# 3. Run
cd /path/to/cuvis.ai.examples/EfficientAD
CUVIS=/lib/cuvis <venv>/bin/python reporting.py -c example_report_config.yaml
```

## Output

`<repo-root>/data/EAD_reporting/test/`:
- `metrics.yaml` — pixel-AUROC, image-AUROC, optimal-Dice + threshold, per-class AUROC
- `AUROC.png`, `AUROC_Class.png`
- `<val>/<stem>.png` — per-frame RGB + SWIR + heatmap + threshold overlays (one per cube)
- `reporting_config.yaml` — config snapshot

## Patches applied vs upstream

See `PATCHES.md` next to this README.
