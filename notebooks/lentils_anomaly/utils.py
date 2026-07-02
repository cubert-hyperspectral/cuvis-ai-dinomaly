"""Shared helpers for the lentils × Dinomaly tutorial notebooks.

Four notebooks ride on these helpers, all on the current **`dinomaly`** plugin
(`cuvis_ai_dinomaly` / `DinomalyDetector`) — never the defunct `dinomaly2`:

- ``lentils_rgb_train_tutorial.ipynb``      — train with the RGB fixed-wavelength selector
- ``lentils_cir_train_tutorial.ipynb``      — train with the CIR (NIR/Red/Green) selector
- ``lentils_concrete_train_tutorial.ipynb`` — train with the concrete band selector
- ``lentils_inference_tutorial.ipynb``      — load a trained pipeline, eval on the 180 test,
                                              per-class AUROC breakdown

Lentils dataset
---------------
61-channel VNIR (430–910 nm), foreign-object anomaly detection. Published on HuggingFace at
``cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils`` (merged cu3s sessions per day +
per-day global COCO). The Dinomaly split (train-on-normals) is the HF ``splits_dinomaly.csv``:
train 308 (normal) / val 148 / test 180 / adaclip_train 500 (held out).

Workflow (per Nima)
-------------------
1. Download cu3s from HF (``huggingface-cli download`` / ``snapshot_download``).
2. Convert to per-frame NPZ with baked ``mask`` + ``class_mask`` via cuvis-ai-dataloader's
   ``cu3s-to-npz`` (never store NPZ on HF).
3. Train / infer from the NPZ via ``MultiNpzDataModule`` (``npz_multi``).

``LENTILS_DATA_SOURCE=local`` (env var, the default here on the dev server) uses the existing
``/mnt/data`` NPZ + the local split CSV; ``=hf`` drives the download→convert path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- config
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLUGINS_YAML = REPO_ROOT / "examples" / "plugins.yaml"

LENTILS_HF_REPO_ID = "cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils"
LENTILS_HF_CACHE = Path.home() / ".cache" / "cuvis_lentils"

#: "local" (default on asai2) reads the existing /mnt/data NPZ + the local split CSV;
#: "hf" downloads cu3s from HF and converts to NPZ. Env-overridable.
LENTILS_DATA_SOURCE = os.environ.get("LENTILS_DATA_SOURCE", "local").lower()

#: The Dinomaly (train-on-normals) split. Local: the npz-path CSV on the dev server.
LOCAL_SPLITS_CSV = Path(
    os.environ.get(
        "LENTILS_SPLITS_CSV",
        "/home/dev/anish/cuvis-ai-dinomaly/diagnostics/lentils_splits_npz_dinomaly.csv",
    )
)
#: Where HF cu3s are downloaded + converted to NPZ (hf mode).
LENTILS_NPZ_OUT = LENTILS_HF_CACHE / "npz"

#: Trained-pipeline dir written by the train notebooks (inference notebook reads it).
LOCAL_PIPELINE_DIR = Path(
    os.environ.get(
        "LENTILS_PIPELINE_DIR",
        str(REPO_ROOT / "notebooks" / "lentils_anomaly" / "outputs" / "trained_run" / "trained_models"),
    )
)

#: COCO category id → name (per-day global COCO; 0 = background/unlabeled).
LENTILS_CATEGORIES: dict[int, str] = {
    0: "Unlabeled", 1: "stem_k", 2: "stone", 3: "alu_shard",
    4: "blue_paper", 5: "white_paper", 6: "fly", 7: "rubber",
}

#: Selector recipes. Each builds the current-dinomaly channel-selector node.
SELECTOR_SPECS: dict[str, dict[str, Any]] = {
    "rgb": {"label": "RGB (650/550/450 nm fixed)", "wavelengths": (650.0, 550.0, 450.0)},
    "cir": {"label": "CIR (NIR 860 / Red 670 / Green 560 nm)",
            "nir_nm": 860.0, "red_nm": 670.0, "green_nm": 560.0},
    "concrete": {"label": "Concrete (learnable band selector)", "output_channels": 3},
}


def resolve_config() -> dict[str, Any]:
    """Notebook-time config (no downloads). Asserts the plugins manifest (ships in the repo)."""
    cfg = {
        "data_source": LENTILS_DATA_SOURCE,
        "hf_repo_id": LENTILS_HF_REPO_ID,
        "plugins_yaml": DEFAULT_PLUGINS_YAML,
        "splits_csv": LOCAL_SPLITS_CSV,
        "npz_out": LENTILS_NPZ_OUT,
        "local_pipeline_dir": LOCAL_PIPELINE_DIR,
        "categories": LENTILS_CATEGORIES,
    }
    assert DEFAULT_PLUGINS_YAML.exists(), (
        f"Plugins manifest not found at {DEFAULT_PLUGINS_YAML}. Run from inside the "
        f"cuvis-ai-dinomaly repo."
    )
    if LENTILS_DATA_SOURCE == "local":
        assert LOCAL_SPLITS_CSV.is_file(), (
            f"LENTILS_DATA_SOURCE='local' but split CSV missing: {LOCAL_SPLITS_CSV}. "
            f"Set LENTILS_DATA_SOURCE='hf' to download + convert from HuggingFace."
        )
    return cfg


# --------------------------------------------------------------------------- selectors
def build_selector(mode: str, *, name: str = "selector") -> Any:
    """Return the channel-selector node for ``mode`` on the current dinomaly stack.

    ``rgb`` -> FixedWavelengthSelector(650/550/450); ``cir`` -> CIRSelector(860/670/560).
    ``concrete`` -> the learnable ConcreteSelector (requires the concrete training branch;
    see the concrete notebook).
    """
    mode = mode.lower()
    if mode == "rgb":
        from cuvis_ai.node.channel_selector import FixedWavelengthSelector

        sel = FixedWavelengthSelector(target_wavelengths=(650.0, 550.0, 450.0), name=name)
    elif mode == "cir":
        from cuvis_ai.node.channel_selector import CIRSelector

        sel = CIRSelector(
            nir_nm=860.0, red_nm=670.0, green_nm=560.0, norm_mode="running",
            running_warmup_frames=0, freeze_running_bounds_after_frames=20, name=name,
        )
    else:
        raise NotImplementedError(
            f"selector mode {mode!r} not wired in this helper; use 'rgb' or 'cir', "
            f"or the concrete training notebook."
        )
    sel._requires_initial_fit_override = False
    return sel


# --------------------------------------------------------------------------- data
def resolve_splits_csv() -> Path:
    """Path to the Dinomaly split CSV the datamodule loads.

    Local: the on-server npz-path CSV. HF: downloads ``splits_dinomaly.csv`` (its npz_path
    column is filled by the convert step — see :func:`ensure_npz`)."""
    if LENTILS_DATA_SOURCE == "local":
        return LOCAL_SPLITS_CSV
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(LENTILS_HF_REPO_ID, repo_type="dataset", filename="splits_dinomaly.csv",
                        cache_dir=str(LENTILS_HF_CACHE))
    )


def load_lentils_frame(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load a per-frame NPZ → ``{cube [H,W,C] f32, wavelengths [C] i32, mask [H,W] i32,
    class_mask [H,W] u8}`` (mask/class_mask zeros when the frame is normal / unbaked)."""
    with np.load(npz_path) as z:
        cube = np.asarray(z["cube"], dtype=np.float32)
        wl = np.asarray(z["wavelengths"]).ravel().astype(np.int32, copy=False)
        h, w = cube.shape[0], cube.shape[1]
        mask = np.asarray(z["mask"], np.int32) if "mask" in z.files else np.zeros((h, w), np.int32)
        class_mask = (
            np.asarray(z["class_mask"], np.uint8) if "class_mask" in z.files
            else np.zeros((h, w), np.uint8)
        )
    return {"cube": cube, "wavelengths": wl, "mask": mask, "class_mask": class_mask}


# --------------------------------------------------------------------------- visualisation
def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else np.clip((x - lo) / (hi - lo), 0, 1)


def false_color(cube_hwc: np.ndarray, wavelengths: np.ndarray, targets_nm: tuple[float, float, float]) -> np.ndarray:
    """Nearest-wavelength 3-channel false-color from a 61-ch cube (for display only)."""
    wl = np.asarray(wavelengths).ravel().astype(float)
    idx = [int(np.argmin(np.abs(wl - t))) for t in targets_nm]
    return _norm(cube_hwc[..., idx])


def render_inference_panel(cube_hwc, score_map, *, wavelengths, gt_mask=None,
                           targets_nm=(650.0, 550.0, 450.0), title=None, figsize=(16.0, 4.0)) -> Any:
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


def per_class_pixel_auroc(scores: list[np.ndarray], class_masks: list[np.ndarray],
                          categories: dict[int, str] | None = None) -> dict[str, float]:
    """One-vs-background pixel AUROC per non-background class (uses the baked ``class_mask``)."""
    from sklearn.metrics import roc_auc_score

    categories = categories or LENTILS_CATEGORIES
    y_s = np.concatenate([_norm(s).ravel() for s in scores]).astype(np.float32)
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


def plot_per_class_auroc_bar(per_class: dict[str, float], *, title="Per-class pixel AUROC",
                             figsize=(10.0, 5.0)) -> Any:
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
