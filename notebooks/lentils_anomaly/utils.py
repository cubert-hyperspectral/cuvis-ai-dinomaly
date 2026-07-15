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
``splits_dinomaly.csv``: train 308 (normal) / val 148 / test 180 / adaclip_train 500 (held out).

Data workflow (the selector split model)
----------------------------------------
1. Download the cu3s dataset from HF via
   :class:`cuvis_ai_core.data.public_datasets.PublicDatasets`.
2. Convert each ``splits_dinomaly.csv`` frame to per-frame NPZ (baked ``mask`` + ``class_mask``)
   with cuvis-ai-dataloader's ``convert_split_manifest``, emitting two artifacts: a
   **universe.csv** (``source, index, path``: the sample universe, one row per frame) and a baked
   **splits.json** (a core ``DataSplitConfig`` of ``file_indices`` selectors). The generated
   splits.json is identical to the dataset's shipped ``splits/dinomaly.json``.
3. Train / infer from the NPZ via ``MultiNpzDataModule`` (``npz_multi``), given the splits.json
   (``DataSplitConfig(splits_path=...)``) resolved over the ``universe_csv``.

:func:`prepare_lentils_data` runs steps 1-2, prefers the dataset's published
``splits/dinomaly.json`` (via :func:`fetch_lentils_splits_json`), and returns
``(splits_json, universe_csv)``.
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
#: The dataset's published baked split (a core DataSplitConfig, train-on-normals) on the Hub.
LENTILS_SPLITS_FILE = "splits/dinomaly.json"

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


def fetch_lentils_splits_json(filename: str = LENTILS_SPLITS_FILE) -> Path:
    """Download the dataset's published split from HuggingFace Hub; return its local cache path.

    ``splits/dinomaly.json`` is the reviewed, position-independent selector split (a core
    ``DataSplitConfig``) published alongside the dataset; it resolves against either the raw cu3s
    sessions or a converted NPZ universe (same ``(source, index)`` identity).
    """
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(LENTILS_HF_REPO_ID, filename, repo_type="dataset"))


def prepare_lentils_data(
    npz_dir: str | Path,
    *,
    dataset_dir: str | Path | None = None,
    limit: int = 0,
    use_published_splits: bool = True,
) -> tuple[Path, Path]:
    """Fetch the lentils cu3s dataset and materialize the NPZ + split artifacts.

    Downloads the dataset from HuggingFace (skipped when already on disk), then converts the
    ``splits_dinomaly.csv`` frames to per-frame NPZ, emitting a ``universe.csv`` (``source, index,
    path``) and a ``splits.json``. When ``use_published_splits`` is set (and this is not a smoke
    run), the dataset's **published** split (``splits/dinomaly.json`` on the Hub) is fetched and
    written over the local ``splits.json``: it is the reviewed, position-independent selector split,
    identical by construction to the regenerated one, and resolves against the ``universe.csv`` here
    (same ``(source, index)`` identity). This pins the split to the published artifact rather than a
    local regenerate, and falls back to the regenerated split when the Hub is unreachable. Returns
    ``(splits_json, universe_csv)``, ready for
    ``MultiNpzDataModule(splits=DataSplitConfig(splits_path=splits_json), universe_csv=universe_csv)``.
    Reruns reuse already-converted frames, so it is cheap to call.

    Parameters
    ----------
    npz_dir
        Where the per-frame NPZ + ``splits.json`` / ``universe.csv`` are written.
    dataset_dir
        Where the raw cu3s dataset is downloaded (default: ``<repo>/../../data``).
    limit
        If > 0, keep at most this many frames per split (fast dry-run); forces the locally
        regenerated split, since the published split spans the full universe.
    use_published_splits
        Prefer the dataset's published ``splits/dinomaly.json`` over the regenerated one.
    """
    import shutil

    from cuvis_ai_core.data.public_datasets import PublicDatasets
    from cuvis_ai_dataloader.data.npz_converter import convert_split_manifest

    npz_dir = Path(npz_dir)
    splits_json = npz_dir / "splits.json"
    universe_csv = npz_dir / "universe.csv"
    dataset_dir = Path(dataset_dir) if dataset_dir else (REPO_ROOT / "data")
    raw_dir = dataset_dir / "XMR_Industrial_Foreign_Object_Detection_Lentils"

    if not (splits_json.is_file() and universe_csv.is_file()):
        PublicDatasets.download_dataset(
            LENTILS_DATASET_NAME, download_path=str(dataset_dir), force=False
        )
        result = convert_split_manifest(
            raw_dir / "splits_dinomaly.csv",
            raw_dir,
            npz_dir,
            universe_csv=universe_csv,
            splits_json=splits_json,
            limit=limit,
        )
        splits_json, universe_csv = result.splits_json, result.universe_csv

    if use_published_splits and not limit:
        try:
            shutil.copyfile(fetch_lentils_splits_json(), splits_json)
        except Exception as exc:  # offline / not yet published: keep the regenerated split
            from loguru import logger

            logger.warning(
                f"Using the regenerated split {splits_json.name}; "
                f"could not fetch the published split: {exc}"
            )
    return splits_json, universe_csv


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
def build_selector(mode: str, *, name: str = "selector") -> Any:
    """Return the channel-selector node for ``mode`` on the current dinomaly stack.

    ``rgb`` -> FixedWavelengthSelector(650/550/450); ``cir`` -> CIRSelector(860/670/560). For the
    AdaCLIP frozen bands, resolve wavelengths with :func:`resolve_adaclip_wavelengths` and build a
    ``FixedWavelengthSelector`` directly (see the adaclip_bands train notebook).
    """
    mode = mode.lower()
    if mode == "rgb":
        from cuvis_ai.node.channel_selector import FixedWavelengthSelector

        sel = FixedWavelengthSelector(target_wavelengths=(650.0, 550.0, 450.0), name=name)
    elif mode == "cir":
        from cuvis_ai.node.channel_selector import CIRSelector

        sel = CIRSelector(
            nir_nm=860.0,
            red_nm=670.0,
            green_nm=560.0,
            norm_mode="running",
            running_warmup_frames=0,
            freeze_running_bounds_after_frames=20,
            name=name,
        )
    else:
        raise NotImplementedError(
            f"selector mode {mode!r} not wired in this helper; use 'rgb' or 'cir'. For the "
            f"AdaCLIP bands, resolve_adaclip_wavelengths(...) + FixedWavelengthSelector (see the "
            f"adaclip_bands train notebook)."
        )
    sel._requires_initial_fit_override = False
    return sel


def resolve_adaclip_wavelengths(
    universe_csv: str | Path, indices: tuple[int, int, int] = ADACLIP_BAND_INDICES
) -> tuple[float, float, float]:
    """Map AdaCLIP's frozen band ``indices`` to wavelengths (nm) using the first universe NPZ.

    Reads the ``path`` of the first row in the ``universe.csv`` (``source, index, path``; the path
    is relative to the CSV), loads its ``wavelengths`` array, and returns
    ``(w[i0], w[i1], w[i2])`` in R,G,B order.
    """
    import csv as _csv

    universe_csv = Path(universe_csv)
    with open(universe_csv, newline="") as f:
        rows = [r for r in _csv.DictReader(f) if (r.get("path") or "").strip()]
    if not rows:
        raise ValueError(f"No rows with a path in {universe_csv}")
    npz_path = Path(rows[0]["path"])
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
