"""Shared helpers for the bedding × Dinomaly tutorial notebooks.

Three notebooks ride on these helpers:

- ``bedding_all6_train_tutorial.ipynb`` — build + train + save the pipeline
- ``bedding_all6_inference_tutorial.ipynb`` — load + run + speedup-recipe demo
- ``bedding_all6_results_tutorial.ipynb`` — render headline + per-class plots

Design notes
------------

The bedding dataset is a 6-channel hyperspectral still-image set (450 / 550 /
625 / 1050 / 1200 / 1450 nm) — semantically different from the lentils tutorial
(61-channel 400–900 nm video). So this util module is purpose-built for
6-channel still frames, not a copy of ``notebooks/lentils_sliding/utils.py``.

Data source (HuggingFace by default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bedding dataset is published on HuggingFace at
``cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding``. By default
the notebooks pull cu3s frames, masks, and ``splits.csv`` straight from HF
(cached under ``BEDDING_HF_CACHE``). Set ``BEDDING_DATA_SOURCE=local`` (env var)
or edit the module constant to read the fast local copy under ``LOCAL_DATA_ROOT``
on the dev server instead.

HF cubes are the **native 2400×4900**; the pretrained pipeline trained on a
center-crop to **1800×4300** (``cube[300:-300, 300:-300]``). ``load_bedding_cube``
applies that crop transparently so HF inference matches the reported metrics
bit-for-bit. The masks get the identical crop.

All path resolution goes through ``load_bedding_cu3s_path`` /
``load_bedding_mask_path`` / ``load_bedding_splits`` / ``load_bedding_cube`` —
notebooks never construct dataset paths directly, so the HF↔local switch is
fully transparent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Configuration — single source of truth for every path the notebooks touch
# ---------------------------------------------------------------------------

#: Wavelengths fed to the 6-channel pipeline, in the order the patch-embed
#: inflation expects (descending λ within each VIS / SWIR triplet, semantically
#: paired so the inflated conv sees matched per-slot statistics).
BEDDING_ALL6_NM: tuple[float, ...] = (625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)

#: Human-readable channel labels matching ``BEDDING_ALL6_NM``.
BEDDING_ALL6_LABELS: tuple[str, ...] = (
    "VIS R (625 nm)", "VIS G (550 nm)", "VIS B (450 nm)",
    "SWIR R (1450 nm)", "SWIR G (1200 nm)", "SWIR B (1050 nm)",
)

#: Path to the saved high-res ALL6 pipeline used by the inference + results
#: notebooks. Trained 20 epochs at 672×672, fp32, RTX A4000.
DEFAULT_PIPELINE_YAML = Path(
    "/mnt/data/cuvis_ai_outputs/dinomaly_bedding_all6_highres_30ep/trained_models/dinomaly_bedding_all6.yaml"
)
DEFAULT_PIPELINE_PT = Path(
    "/mnt/data/cuvis_ai_outputs/dinomaly_bedding_all6_highres_30ep/trained_models/dinomaly_bedding_all6.pt"
)

#: Path to the saved per-pilot eval outputs (used by the results notebook).
DEFAULT_EVAL_DIR = Path(
    "/mnt/data/cuvis_ai_outputs/dinomaly_bedding_all6_highres_30ep/eval_val"
)

#: Plugins manifest registering the dinomaly + bedding nodes.
DEFAULT_PLUGINS_YAML = Path("/home/dev/anish/cuvis-ai-dinomaly/examples/plugins.yaml")

#: Splits CSV (train / val) describing the bedding dataset NPZ frames (local mode).
DEFAULT_SPLITS_CSV = Path("/mnt/data/bedding_dataset_npz/bedding_splits_npz.csv")

#: Local-disk root for the cu3s session files (raw hyperspectral cubes, local mode).
DEFAULT_CU3S_VAL_ROOT = Path("/mnt/data/bedding_dataset/exported/val")

#: Mask PNG root used by the EAD example's reporting path. Notebooks use this
#: for the GT overlay when rendering qualitative results (local mode).
DEFAULT_MASK_ROOT = Path("/mnt/data/bedding_dataset/labels_extracted/labels")

#: HuggingFace dataset repo id. The published bedding dataset:
#: https://huggingface.co/datasets/cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding
BEDDING_HF_REPO_ID = "cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding"
BEDDING_HF_CACHE = Path.home() / ".cache" / "cuvis_bedding"

#: Data source: "hf" (default) downloads from BEDDING_HF_REPO_ID and caches under
#: BEDDING_HF_CACHE; "local" reads from LOCAL_DATA_ROOT (fast, dev-server only).
#: Override per-session with the ``BEDDING_DATA_SOURCE`` env var or edit here.
BEDDING_DATA_SOURCE = os.environ.get("BEDDING_DATA_SOURCE", "hf").lower()

#: Local dataset root, used only when ``BEDDING_DATA_SOURCE == "local"``.
LOCAL_DATA_ROOT = Path("/mnt/data/bedding_dataset")

#: Center-crop applied to native HF cubes (2400×4900) to match the pipeline's
#: training crop (1800×4300). Mirrors ``EAD_CROP`` in convert_bedding_cu3s_to_npz.py.
_TRAINING_CROP = (slice(300, -300), slice(300, -300))


def resolve_default_config() -> dict[str, Any]:
    """Resolve every notebook-time path in one dict.

    Notebooks call this once at the top and pass the resulting ``config`` dict
    around. The model artefacts (pipeline YAML/PT, plugins manifest) are always
    asserted to exist — they are NOT on HuggingFace and must be produced by the
    training notebook or otherwise present locally. The *dataset* paths are only
    asserted in ``local`` mode; in the default ``hf`` mode they are resolved on
    demand by the loaders below (and downloaded + cached transparently).
    """
    cfg: dict[str, Any] = {
        "data_source": BEDDING_DATA_SOURCE,
        "hf_repo_id": BEDDING_HF_REPO_ID,
        "pipeline_yaml": DEFAULT_PIPELINE_YAML,
        "pipeline_pt": DEFAULT_PIPELINE_PT,
        "eval_dir": DEFAULT_EVAL_DIR,
        "plugins_yaml": DEFAULT_PLUGINS_YAML,
        "splits_csv": DEFAULT_SPLITS_CSV,
        "cu3s_val_root": DEFAULT_CU3S_VAL_ROOT,
        "mask_root": DEFAULT_MASK_ROOT,
        "bedding_all6_nm": BEDDING_ALL6_NM,
        "bedding_all6_labels": BEDDING_ALL6_LABELS,
    }
    # Model artefacts must always exist (not distributed via HF).
    for key in ("pipeline_yaml", "pipeline_pt", "plugins_yaml"):
        assert cfg[key].exists(), f"Missing required path: cfg[{key!r}] = {cfg[key]}"
    # Dataset paths are only required when reading from the local mount.
    if BEDDING_DATA_SOURCE == "local":
        for key in ("splits_csv", "cu3s_val_root"):
            assert cfg[key].exists(), (
                f"BEDDING_DATA_SOURCE='local' but missing {key!r} = {cfg[key]}. "
                f"Set BEDDING_DATA_SOURCE='hf' to download from HuggingFace instead."
            )
    return cfg


# ---------------------------------------------------------------------------
# Data loaders — HuggingFace by default, local mount via BEDDING_DATA_SOURCE
# ---------------------------------------------------------------------------

def center_crop_to_training(arr: np.ndarray) -> np.ndarray:
    """Center-crop a native ``2400×4900`` array to the ``1800×4300`` training crop.

    Works for cubes ``[H, W, C]`` and masks ``[H, W]``. Mirrors the
    ``cube[300:-300, 300:-300]`` slice in
    ``convert_bedding_cu3s_to_npz.py`` (the EAD-style center crop the pretrained
    pipeline was trained on). A no-op short-circuit handles arrays that are
    already cropped (local mode), so it is safe to call unconditionally.
    """
    if arr.shape[0] == 1800 and arr.shape[1] == 4300:
        return arr  # already cropped (local NPZ / cu3s)
    return arr[_TRAINING_CROP]


def _hf_download(filename: str) -> Path:
    """Download ``filename`` from the HF dataset repo, cached, return its path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - dependency present in env
        raise RuntimeError(
            "HF data source requires `huggingface_hub` (pip install huggingface_hub)."
        ) from e
    return Path(
        hf_hub_download(
            repo_id=BEDDING_HF_REPO_ID,
            repo_type="dataset",
            filename=filename,
            cache_dir=str(BEDDING_HF_CACHE),
        )
    )


def load_bedding_cu3s_path(frame_stem: str, *, split: str = "val") -> Path:
    """Resolve a frame stem (without ``.cu3s``) to an on-disk cu3s file.

    HF mode (default): downloads ``data/{split}/{stem}.cu3s`` and returns the
    cached path. Local mode: returns ``LOCAL_DATA_ROOT/exported/{split}/{stem}.cu3s``.
    """
    if BEDDING_DATA_SOURCE == "local":
        return LOCAL_DATA_ROOT / "exported" / split / f"{frame_stem}.cu3s"
    return _hf_download(f"data/{split}/{frame_stem}.cu3s")


def load_bedding_mask_path(frame_stem: str) -> Path | None:
    """Resolve a frame stem to its GT mask PNG, or ``None`` if absent.

    Mirrors ``find_mask_png`` in convert_bedding_cu3s_to_npz.py: prefer the
    cube-side ``{stem}_mask.png``, fall back to the RGB-side
    ``{stem}_{stem}_(0000|0)_RGB_mask.png``. Normal (all-background) frames have
    no mask — callers treat ``None`` as an empty mask. Returned masks are native
    2400×4900 and must be center-cropped by the caller (``center_crop_to_training``).
    """
    candidates = (
        f"{frame_stem}_mask.png",
        f"{frame_stem}_{frame_stem}_0000_RGB_mask.png",
        f"{frame_stem}_{frame_stem}_0_RGB_mask.png",
    )
    if BEDDING_DATA_SOURCE == "local":
        for name in candidates:
            p = DEFAULT_MASK_ROOT / name
            if p.is_file():
                return p
        return None
    for name in candidates:
        try:
            return _hf_download(f"annotations_raw/labels/{name}")
        except Exception:
            continue  # EntryNotFoundError — try the next naming pattern
    return None


def load_bedding_splits():
    """Return the dataset splits as a pandas DataFrame.

    HF mode: downloads the root ``splits.csv`` (cols: split, stem, cu3s_path,
    coco_json_path, image_id, filename_label, has_annotation, category_ids,
    label_fault). Local mode: reads the NPZ splits CSV. In both cases the result
    exposes at least ``split`` and ``stem`` so notebooks can iterate frames.
    """
    import pandas as pd

    if BEDDING_DATA_SOURCE == "local":
        df = pd.read_csv(DEFAULT_SPLITS_CSV)
        # Local NPZ CSV uses npz_path; derive a stem column if missing.
        if "stem" not in df.columns and "npz_path" in df.columns:
            df["stem"] = df["npz_path"].map(lambda p: Path(str(p)).stem)
        return df
    return pd.read_csv(_hf_download("splits.csv"))


def load_bedding_cube(frame_stem: str, *, split: str = "val") -> tuple[np.ndarray, np.ndarray]:
    """Load a frame's cube + wavelengths, center-cropped to the training size.

    Opens the cu3s via ``cuvis.SessionFile`` (downloading from HF first in the
    default mode), returns ``(cube_hwc_float32_1800x4300x6, wavelengths_nm)``.
    The crop is applied so the cube matches exactly what the pretrained pipeline
    saw at train time — HF (native 2400×4900) and local (pre-cropped) paths
    therefore produce bit-identical model inputs.
    """
    import cuvis

    cu3s_path = load_bedding_cu3s_path(frame_stem, split=split)
    assert cu3s_path.is_file(), f"cu3s not found: {cu3s_path}"
    mesu = cuvis.SessionFile(str(cu3s_path)).get_measurement(0)
    cube = np.asarray(mesu.data["cube"].array, dtype=np.float32)
    wavelengths = np.asarray(mesu.data["cube"].wavelength, dtype=np.int32)
    return center_crop_to_training(cube), wavelengths


# ---------------------------------------------------------------------------
# 6-channel visualisation helpers
# ---------------------------------------------------------------------------

def split_cube_vis_swir(cube_bhwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 6-channel cube ``[B, H, W, 6]`` into VIS (R/G/B at 625/550/450 nm)
    and SWIR-as-pseudo-RGB (R/G/B at 1450/1200/1050 nm) views.

    The output channel order matches ``BEDDING_ALL6_NM`` — long-λ first within
    each triplet so the displayed VIS/SWIR triplets honour the
    descending-λ→R/G/B convention.

    Both outputs are returned in ``[H, W, 3]`` shape (batch dim squeezed for
    visualisation; if B>1 only the first item is used).
    """
    assert cube_bhwc.ndim == 4 and cube_bhwc.shape[-1] == 6, \
        f"expected [B,H,W,6] cube, got {cube_bhwc.shape}"
    vis = cube_bhwc[0, ..., :3]
    swir = cube_bhwc[0, ..., 3:]
    return vis, swir


def normalize_for_display(x: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1] for matplotlib imshow.

    Robust to all-zero arrays (returns zeros) and clips floats.
    """
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def render_input_triplets(
    cube_bhwc: np.ndarray,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (12.0, 4.5),
):
    """Render side-by-side VIS-RGB and SWIR-pseudo-RGB views of a 6-ch cube.

    This is the bedding analog of the lentils notebook's RGB / CIR / custom
    triptych — but for 6 channels (3 VIS + 3 SWIR) instead of 61.
    """
    vis, swir = split_cube_vis_swir(cube_bhwc)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(normalize_for_display(vis))
    axes[0].set_title("VIS (625 / 550 / 450 nm)")
    axes[0].axis("off")
    axes[1].imshow(normalize_for_display(swir))
    axes[1].set_title("SWIR (1450 / 1200 / 1050 nm)")
    axes[1].axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def render_inference_panel(
    cube_bhwc: np.ndarray,
    score_map: np.ndarray,
    *,
    gt_mask: np.ndarray | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (16.0, 4.0),
):
    """Render the per-frame qualitative story: VIS, SWIR, score heatmap, GT overlay.

    Parameters
    ----------
    cube_bhwc : np.ndarray
        Input cube ``[B, H, W, 6]`` (batch=1 OK; first item is used).
    score_map : np.ndarray
        Pixel-wise anomaly score, ``[H, W]`` or ``[1, H, W, 1]``.
    gt_mask : np.ndarray, optional
        Binary ground-truth mask, same H×W as the score map.
    """
    if score_map.ndim == 4:
        score_map = score_map[0, ..., 0]
    elif score_map.ndim == 3 and score_map.shape[-1] == 1:
        score_map = score_map[..., 0]
    score_disp = normalize_for_display(score_map)

    n_cols = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    vis, swir = split_cube_vis_swir(cube_bhwc)
    axes[0].imshow(normalize_for_display(vis))
    axes[0].set_title("VIS")
    axes[0].axis("off")
    axes[1].imshow(normalize_for_display(swir))
    axes[1].set_title("SWIR")
    axes[1].axis("off")
    axes[2].imshow(score_disp, cmap="inferno")
    axes[2].set_title("Score (min-max norm.)")
    axes[2].axis("off")

    if gt_mask is not None:
        if gt_mask.ndim == 4:
            gt_mask = gt_mask[0, ..., 0]
        elif gt_mask.ndim == 3 and gt_mask.shape[-1] == 1:
            gt_mask = gt_mask[..., 0]
        # VIS underlay with GT contour overlay
        axes[3].imshow(normalize_for_display(vis))
        axes[3].contour(gt_mask > 0, levels=[0.5], colors="red", linewidths=1.5)
        axes[3].set_title("VIS + GT contour")
        axes[3].axis("off")

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Headline + per-class plotting helpers (used by the results notebook)
# ---------------------------------------------------------------------------

def plot_per_class_auroc_bar(
    per_class_json_path: Path,
    *,
    title: str = "Per-class pixel AUROC (EAD methodology)",
    figsize: tuple[float, float] = (12.0, 6.0),
):
    """Render a horizontal bar chart of per-class AUROC from a recomputed json."""
    import json
    data = json.loads(Path(per_class_json_path).read_text())
    items = data["per_class_auroc"] if "per_class_auroc" in data else data
    names = list(items.keys())
    aurocs = [items[n]["auroc"] if isinstance(items[n], dict) else items[n] for n in names]
    order = np.argsort(aurocs)
    names_sorted = [names[i] for i in order]
    aurocs_sorted = [aurocs[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(names_sorted, aurocs_sorted, color="steelblue")
    ax.set_xlim(0.5, 1.01)
    ax.axvline(1.0, color="black", linewidth=0.5)
    ax.set_xlabel("Pixel AUROC")
    ax.set_title(title)
    for bar, val in zip(bars, aurocs_sorted):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def load_headline_report(report_json_path: Path) -> dict[str, Any]:
    """Load eval_val/report.json. Used by the results notebook header."""
    import json
    return json.loads(Path(report_json_path).read_text())


# ---------------------------------------------------------------------------
# Inference helpers — speedup recipe demo (mirrors verify_fast_inference_metrics)
# ---------------------------------------------------------------------------

def apply_lossless_speedups(pipeline, *, autocast_dtype: torch.dtype = torch.bfloat16,
                            compile_mode: str = "reduce-overhead"):
    """Enable TF32 + bf16 autocast + ``torch.compile`` on the underlying model.

    Returns the matching ``torch.autocast`` context manager. Callers should run
    inference inside both ``torch.inference_mode()`` and the returned context.

    The recipe is verified lossless: pixel AUROC matches fp32 to 5 decimals,
    image AUROC within ±0.003, Dice within ±0.0002 on the bedding val set.
    """
    torch.set_float32_matmul_precision("high")
    for _name, mod in pipeline.torch_layers.named_children():
        if hasattr(mod, "dinomaly_model"):
            mod.dinomaly_model = torch.compile(mod.dinomaly_model, mode=compile_mode)
    return torch.autocast(device_type="cuda", dtype=autocast_dtype)
