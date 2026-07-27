"""Shared helpers for the lentils × Dinomaly tutorial notebooks.

Four notebooks ride on these helpers, all on the current **`dinomaly`** plugin
(`cuvis_ai_dinomaly` / `DinomalyDetector`) -- never the defunct `dinomaly2`:

- ``lentils_rgb_train_tutorial.ipynb``          -- train with the RGB fixed-wavelength selector
- ``lentils_cir_train_tutorial.ipynb``          -- train with the CIR (NIR/Red/Green) selector
- ``lentils_adaclip_bands_train_tutorial.ipynb``-- train with the 3 bands AdaCLIP's frozen concrete
                                                  selector converged to (indices 14/59/57), fixed
- ``lentils_inference_tutorial.ipynb``          -- load a trained pipeline, eval on the 180 test,
                                                  per-class AUROC breakdown

Lentils dataset
---------------
61-channel VNIR (430-910 nm), foreign-object anomaly detection. Published on HuggingFace at
``cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils`` (merged cu3s sessions per day +
per-day global COCO). The Dinomaly split (train-on-normals) is the dataset's
``splits/dinomaly.json`` (a selector over the shipped ``universe.csv``): train 308 (normal) /
val 148 / test 180 (adaclip_train 500 held out in ``splits/adaclip.json``).

Data workflow (the selector split model)
----------------------------------------
1. Download the cu3s dataset from HF via
   :class:`cuvis_ai_core.data.public_datasets.PublicDatasets`.
2. Convert the frames ``splits/dinomaly.json`` selects to per-frame NPZ (baked ``mask`` +
   ``class_mask``) with cuvis-ai-dataloader's ``convert_universe``, emitting two artifacts: a
   **universe.csv** (``source, index, materialized_path``: the sample universe, one row per frame)
   and a baked **splits.json** (a core ``DataSplitConfig`` of ``file_indices`` selectors). The
   generated splits.json is identical to the dataset's shipped ``splits/dinomaly.json``.
3. Train / infer from the NPZ via ``MultiNpzDataModule`` (``npz_multi``), given the splits.json
   (``DataSplitConfig(splits_path=...)``) resolved over the ``universe_csv``.

The notebooks run steps 1-2 directly with ``PublicDatasets.download_dataset`` +
``convert_universe``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- config
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLUGINS_YAML = REPO_ROOT / "examples" / "plugins.yaml"

LENTILS_HF_REPO_ID = "cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils"
#: PublicDatasets registry name / alias for the dataset (see cuvis-ai-core public_datasets).
LENTILS_DATASET_NAME = "industrial_fod_lentils"

#: Trained-pipeline dir the inference notebook reads by default (the RGB train notebook's output).
#: Pass a different dir to :func:`resolve_pipeline` to evaluate a CIR / AdaCLIP-bands run.
DEFAULT_PIPELINE_DIR = (
    REPO_ROOT / "notebooks" / "lentils_anomaly" / "outputs" / "lentils_rgb_run" / "trained_models"
)

#: COCO category id -> name (per-day global COCO; 0 = background/unlabeled).
LENTILS_CATEGORIES: dict[int, str] = {
    0: "Unlabeled",
    1: "stem_k",
    2: "stone",
    3: "alu_shard",
    4: "blue_paper",
    5: "white_paper",
    6: "fly",
    7: "rubber",
}

#: The three cube-channel indices AdaCLIP's frozen concrete selector converged to on lentils.
#: Resolved to wavelengths per-dataset by :func:`resolve_adaclip_wavelengths` (typically
#: ~(542, 902, 886) nm on the 61-band lentils data), then fed to a FixedWavelengthSelector.
ADACLIP_BAND_INDICES: tuple[int, int, int] = (14, 59, 57)


def resolve_config() -> dict[str, Any]:
    """Notebook-time config (no downloads). Asserts the plugins manifest (ships in the repo)."""
    cfg = {
        "hf_repo_id": LENTILS_HF_REPO_ID,
        "dataset_name": LENTILS_DATASET_NAME,
        "plugins_yaml": DEFAULT_PLUGINS_YAML,
        "default_pipeline_dir": DEFAULT_PIPELINE_DIR,
        "categories": LENTILS_CATEGORIES,
    }
    assert DEFAULT_PLUGINS_YAML.exists(), (
        f"Plugins manifest not found at {DEFAULT_PLUGINS_YAML}. Run from inside the "
        f"cuvis-ai-dinomaly repo."
    )
    return cfg


def resolve_pipeline(pipeline_dir: str | Path = DEFAULT_PIPELINE_DIR) -> tuple[Path, Path]:
    """Return ``(yaml_path, pt_path)`` for a trained lentils pipeline.

    Picks the single ``*.yaml`` in ``pipeline_dir`` (a train notebook's ``trained_models`` dir) +
    its sibling ``.pt``. Pass a specific run's dir (RGB / CIR / AdaCLIP-bands) to choose.
    """
    d = Path(pipeline_dir)
    yamls = sorted(d.glob("*.yaml"))
    if not yamls:
        raise FileNotFoundError(
            f"No trained pipeline *.yaml in {d}. Run a train notebook first, or point "
            f"PIPELINE_DIR (setup cell) at a trained_models dir."
        )
    yaml_path = yamls[0]
    pt_path = yaml_path.with_suffix(".pt")
    if not pt_path.is_file():
        raise FileNotFoundError(f"Missing weights next to {yaml_path.name}: {pt_path}")
    return yaml_path, pt_path


# --------------------------------------------------------------------------- selectors
def resolve_adaclip_wavelengths(
    universe_csv: str | Path, indices: tuple[int, int, int] = ADACLIP_BAND_INDICES
) -> tuple[float, float, float]:
    """Map AdaCLIP's frozen band ``indices`` to wavelengths (nm) using the first universe NPZ.

    Reads the ``materialized_path`` of the first row in the ``universe.csv`` (``source, index,
    materialized_path``; the path is relative to the CSV), loads its ``wavelengths`` array, and
    returns ``(w[i0], w[i1], w[i2])`` in R,G,B order.
    """
    import csv as _csv

    universe_csv = Path(universe_csv)
    with open(universe_csv, newline="") as f:
        rows = [r for r in _csv.DictReader(f) if (r.get("materialized_path") or "").strip()]
    if not rows:
        raise ValueError(f"No rows with a materialized_path in {universe_csv}")
    npz_path = Path(rows[0]["materialized_path"])
    if not npz_path.is_absolute():
        npz_path = (universe_csv.parent / npz_path).resolve()
    with np.load(npz_path) as z:
        if "wavelengths" not in z.files:
            raise KeyError(f"{npz_path} has no 'wavelengths'")
        w = np.asarray(z["wavelengths"], dtype=np.float64).ravel()
    for i in indices:
        if not (0 <= i < len(w)):
            raise IndexError(f"band index {i} out of range for {len(w)} bands")
    return (float(w[indices[0]]), float(w[indices[1]]), float(w[indices[2]]))


# --------------------------------------------------------------------------- data
def load_lentils_frame(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load a per-frame NPZ -> ``{cube [H,W,C] f32, wavelengths [C] i32, mask [H,W] i32,
    class_mask [H,W] u8}`` (mask/class_mask zeros when the frame is normal / unbaked)."""
    with np.load(npz_path) as z:
        cube = np.asarray(z["cube"], dtype=np.float32)
        wl = np.asarray(z["wavelengths"]).ravel().astype(np.int32, copy=False)
        h, w = cube.shape[0], cube.shape[1]
        mask = np.asarray(z["mask"], np.int32) if "mask" in z.files else np.zeros((h, w), np.int32)
        class_mask = (
            np.asarray(z["class_mask"], np.uint8)
            if "class_mask" in z.files
            else np.zeros((h, w), np.uint8)
        )
    return {"cube": cube, "wavelengths": wl, "mask": mask, "class_mask": class_mask}


# --------------------------------------------------------------------------- visualisation
def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else np.clip((x - lo) / (hi - lo), 0, 1)


def false_color(
    cube_hwc: np.ndarray, wavelengths: np.ndarray, targets_nm: tuple[float, float, float]
) -> np.ndarray:
    """Nearest-wavelength 3-channel false-color from a 61-ch cube (for display only)."""
    wl = np.asarray(wavelengths).ravel().astype(float)
    idx = [int(np.argmin(np.abs(wl - t))) for t in targets_nm]
    return _norm(cube_hwc[..., idx])


def render_inference_panel(
    cube_hwc,
    score_map,
    *,
    wavelengths,
    gt_mask=None,
    targets_nm=(650.0, 550.0, 450.0),
    title=None,
    figsize=(16.0, 4.0),
) -> Any:
    """Per-frame story: false-color RGB, anomaly heatmap, and GT contour overlay."""
    if score_map.ndim == 4:
        score_map = score_map[0, ..., 0]
    elif score_map.ndim == 3 and score_map.shape[-1] == 1:
        score_map = score_map[..., 0]
    rgb = false_color(cube_hwc, wavelengths, targets_nm)
    n = 3 if gt_mask is not None else 2
    fig, ax = plt.subplots(1, n, figsize=figsize)
    ax[0].imshow(rgb)
    ax[0].set_title("false-color RGB")
    ax[0].axis("off")
    ax[1].imshow(_norm(score_map), cmap="inferno")
    ax[1].set_title("anomaly score")
    ax[1].axis("off")
    if gt_mask is not None:
        if gt_mask.ndim > 2:
            gt_mask = np.squeeze(gt_mask)
        ax[2].imshow(rgb)
        ax[2].contour(gt_mask > 0, levels=[0.5], colors="red", linewidths=1.2)
        ax[2].set_title("RGB + GT contour")
        ax[2].axis("off")
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def panel_frames(collected: list, datamodule: Any) -> list[dict[str, Any]]:
    """Flatten a ``Predictor`` collect run into one slim record per frame for the panels.

    Pulls the anomaly ``scores`` map, binary ``mask``, and per-frame ``anomaly_score`` out of each
    per-batch ``(node, port)`` output dict (already moved to CPU by ``collect_ports``), plus the
    frame's NPZ ``path`` / ``index`` from the predict dataset. The ground-truth cube for false-color
    rendering is reloaded from that path by the caller; per-class AUROC comes from the metric node,
    so neither the cube nor the multi-class mask is flattened here.
    """

    def _port(batch_out: dict, port: str) -> Any:
        for (_node, name), value in batch_out.items():
            if name == port and value is not None:
                return value
        return None

    def _frame(x: Any, i: int) -> np.ndarray | None:
        return None if x is None else x[i].detach().float().cpu().numpy()

    records = getattr(datamodule.predict_ds, "records", None) or getattr(
        datamodule.predict_ds, "_rows", None
    )
    frames: list[dict[str, Any]] = []
    offset = 0
    for batch_out in collected:
        scores = _port(batch_out, "scores")
        ascore = _port(batch_out, "anomaly_score")
        mask = _port(batch_out, "mask")
        bsz = int((scores if scores is not None else mask).shape[0])
        for i in range(bsz):
            rec = records[offset + i] if records is not None and offset + i < len(records) else {}
            score = _frame(scores, i)
            if score is not None and score.ndim == 3 and score.shape[-1] == 1:
                score = score[..., 0]
            m = _frame(mask, i)
            frames.append(
                {
                    "score_map": None if score is None else score.astype(np.float32),
                    "anomaly_score": None if ascore is None else float(ascore[i].item()),
                    "mask": None if m is None else m.astype(np.int32),
                    "path": rec.get("path") if isinstance(rec, dict) else None,
                    "index": rec.get("index") if isinstance(rec, dict) else None,
                    "is_anomalous": bool(m is not None and m.any()),
                }
            )
        offset += bsz
    return frames


def per_class_pixel_auroc(
    scores: list[np.ndarray],
    class_masks: list[np.ndarray],
    categories: dict[int, str] | None = None,
) -> dict[str, float]:
    """One-vs-background pixel AUROC per non-background class (uses the baked ``class_mask``).

    Pools **raw** scores across frames -- AUROC is rank-based, so per-frame min-max normalization
    would not be monotonic across frames and would distort the pooled ranking.
    """
    from sklearn.metrics import roc_auc_score

    categories = categories or LENTILS_CATEGORIES
    y_s = np.concatenate([np.asarray(s, np.float32).ravel() for s in scores])
    cm = np.concatenate([np.asarray(c).ravel() for c in class_masks])
    out: dict[str, float] = {}
    for cid, cname in categories.items():
        if cid == 0:
            continue
        cls_pixels = cm == cid
        if not cls_pixels.any():
            continue
        # one-vs-background: positives = this class, negatives = background (exclude other classes)
        keep = cls_pixels | (cm == 0)
        out[cname] = float(roc_auc_score(cls_pixels[keep].astype(int), y_s[keep]))
    return out


def plot_per_class_auroc_bar(
    per_class: dict[str, float], *, title="Per-class pixel AUROC", figsize=(10.0, 5.0)
) -> Any:
    if not per_class:
        return None
    names = sorted(per_class, key=per_class.get)
    vals = [per_class[n] for n in names]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(names, vals, color="steelblue")
    ax.set_xlim(0.5, 1.01)
    ax.axvline(1.0, color="black", lw=0.5)
    ax.set_xlabel("Pixel AUROC")
    ax.set_title(title)
    for b, v in zip(bars, vals, strict=True):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    return fig
