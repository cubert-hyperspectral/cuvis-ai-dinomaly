# Patches applied to upstream `cuvis.ai.examples/EfficientAD/reporting.py`

All five live in the vendored `reporting.py` in this folder. Upstream md5 at
time of vendoring: see git blame on `reporting.py` in `cuvis.ai.examples`
(commit hash should be recorded in the repo when reproducing).

## 1. `labels` key in config made optional (line ~228)
Upstream assumes every config has a `labels:` list. Bedding doesn't supply one
(masks live next to cubes via symlink).

```python
# before
for dataset_path, labels_path in zip(config['datasets'], config['labels']):
# after
for dataset_path, labels_path in zip(
        config['datasets'],
        config.get('labels', [None] * len(config['datasets']))):
```

## 2. Flat cu3s glob (line ~231)
Upstream assumes `<dataset>/<class>/<cube>.cu3s` (MVTec layout). Bedding's
`exported/val/` is flat.

```python
# before
cubes = glob.glob(str(data_path / "*" / "*.cu3s"))
# after
cubes = glob.glob(str(data_path / "*.cu3s")) or glob.glob(str(data_path / "*" / "*.cu3s"))
```

## 3. Per-class ROC no longer drops multi-class frames (line ~162)
Upstream filter `if len(np.unique(gt_mask)) > 2: continue` silently dropped
every frame whose mask had more than one foreground class — i.e. almost every
bedding frame. Result: only "water" was scored. Removed the filter; one-vs-rest
is correct for multi-class frames too.

## 4. Replaced `torchmetrics.functional.dice` + added image-AUROC (line ~275)
`torchmetrics.functional.dice` was removed in torchmetrics ≥ 1.0. Replaced
with **optimal-F1 Dice** (max F1 across PR-curve thresholds) and added
**image-level AUROC** (max anomaly score per frame vs binary frame label).

```python
from sklearn.metrics import precision_recall_curve, roc_auc_score
scores_flat = np.concatenate([x.ravel() for x in all_scores]).astype(np.float32)
gt_flat = np.concatenate([x.ravel().astype(np.uint8) for x in binary_truths])
precision, recall, thresholds = precision_recall_curve(gt_flat, scores_flat)
f1 = 2 * precision * recall / (precision + recall + 1e-12)
best_idx = int(np.argmax(f1))
dice_score = float(f1[best_idx])
best_threshold = float(thresholds[min(best_idx, len(thresholds)-1)])

img_scores = np.array([float(x.max()) for x in all_scores])
img_labels = np.array([int(m.any()) for m in binary_truths])
image_auroc = float(roc_auc_score(img_labels, img_scores))
```

The `metrics[dataset_name]` dict now also carries `n_frames`, `n_frames_with_gt`,
`dice_threshold`, and `image_auroc` for transparency.

## 5. Deterministic seeding (line ~395)
Upstream relied on `seed: 42` being in the config but never actually called
the seed functions. Added explicit RNG seeding + cuDNN determinism flags +
`Trainer(deterministic=True)`.
