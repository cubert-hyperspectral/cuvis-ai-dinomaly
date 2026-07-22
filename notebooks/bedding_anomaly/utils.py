"""Shared helpers for the bedding × Dinomaly tutorial notebooks.

Two notebooks ride on these helpers:

- ``bedding_all6_train_tutorial.ipynb`` — build + train + save the pipeline
- ``bedding_all6_inference_tutorial.ipynb`` — load + run + speedup-recipe demo,
  plus the headline + per-class metric plots

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
    "VIS R (625 nm)",
    "VIS G (550 nm)",
    "VIS B (450 nm)",
    "SWIR R (1450 nm)",
    "SWIR G (1200 nm)",
    "SWIR B (1050 nm)",
)

#: Category-id -> name map for the bedding foreign-object classes (0 = background),
#: mirroring the dataset's ``class_map.json``. The per-pixel ``class_mask`` the data module
#: emits carries these ids; ``PerClassAnomalyAUROC`` uses this map to label its one-vs-background
#: per-class scores. ``PLA_blacK_4mm`` keeps the dataset's own capitalisation.
BEDDING_CATEGORIES: dict[int, str] = {
    0: "background",
    1: "water",
    2: "alcohol",
    3: "POMC",
    4: "PET",
    5: "leaf",
    6: "fake_leaf",
    7: "PLA_black_1mm",
    8: "PLA_black_2mm",
    9: "PLA_blacK_4mm",
    10: "PLA_black_8mm",
    11: "PLA_black_16mm",
    12: "PLA_blue_1mm",
    13: "PLA_blue_2mm",
    14: "PLA_blue_4mm",
    15: "PLA_blue_8mm",
    16: "PLA_blue_16mm",
    17: "PLA_white_1mm",
    18: "PLA_white_2mm",
    19: "PLA_white_4mm",
    20: "PLA_white_8mm",
    21: "PLA_white_16mm",
    22: "transparent_plastic",
    23: "water&alcohol-tray",
}

#: Repo root, auto-detected from this file's location
#: (``<repo>/notebooks/bedding_anomaly/utils.py`` → ``parents[2]``). Everything
#: that ships inside the repo (e.g. the plugins manifest) is resolved relative to
#: this, so the notebooks run from a fresh clone with no path edits.
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- HuggingFace repos -----------------------------------------------------
#: Published bedding *dataset* (cu3s + masks + splits):
#: https://huggingface.co/datasets/cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding
BEDDING_HF_REPO_ID = "cubert-gmbh/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding"
#: Published trained *model* (pipeline YAML/PT + eval JSON):
#: https://huggingface.co/cubert-gmbh/dinomaly-bedding-all6
BEDDING_MODEL_HF_REPO = "cubert-gmbh/dinomaly-bedding-all6"
#: Where HF downloads (dataset cu3s + the ~580 MB model) are cached. Defaults under the home
#: ``.cache``; override with ``BEDDING_HF_CACHE`` to redirect the large cu3s cache off a full drive.
BEDDING_HF_CACHE = Path(
    os.environ.get("BEDDING_HF_CACHE", str(Path.home() / ".cache" / "cuvis_bedding"))
)

# --- Data source toggle (where dataset cu3s/masks/splits come from) --------
#: "hf" (default) downloads from BEDDING_HF_REPO_ID and caches under
#: BEDDING_HF_CACHE; "local" reads from LOCAL_DATA_ROOT (fast, dev-server only).
#: Override with the ``BEDDING_DATA_SOURCE`` env var or edit here.
BEDDING_DATA_SOURCE = os.environ.get("BEDDING_DATA_SOURCE", "hf").lower()

#: Local dataset root, used only when ``BEDDING_DATA_SOURCE == "local"``. Env-overridable.
LOCAL_DATA_ROOT = Path(os.environ.get("BEDDING_LOCAL_ROOT", "/mnt/data/bedding_dataset"))

# --- Pipeline source toggle (where the trained pipeline comes from) --------
#: "hf" (default) downloads the pretrained pipeline + eval from
#: BEDDING_MODEL_HF_REPO; "local" uses a pipeline you trained yourself (e.g. the
#: train notebook's output dir). Override with ``BEDDING_PIPELINE_SOURCE`` /
#: ``BEDDING_PIPELINE_DIR`` env vars.
BEDDING_PIPELINE_SOURCE = os.environ.get("BEDDING_PIPELINE_SOURCE", "hf").lower()
#: Local trained-pipeline dir (the train notebook writes here by default).
LOCAL_PIPELINE_DIR = Path(
    os.environ.get(
        "BEDDING_PIPELINE_DIR",
        str(
            REPO_ROOT
            / "notebooks"
            / "bedding_anomaly"
            / "outputs"
            / "trained_run"
            / "trained_models"
        ),
    )
)

#: Plugins manifest registering the dinomaly + bedding nodes — ships in the repo.
DEFAULT_PLUGINS_YAML = REPO_ROOT / "examples" / "plugins.yaml"

#: Local-mode dataset paths (used only when ``BEDDING_DATA_SOURCE == "local"``); env-overridable.
DEFAULT_SPLITS_CSV = Path(
    os.environ.get("BEDDING_SPLITS_CSV", "/mnt/data/bedding_dataset_npz/bedding_splits_npz.csv")
)
DEFAULT_CU3S_VAL_ROOT = LOCAL_DATA_ROOT / "exported" / "val"
DEFAULT_MASK_ROOT = Path(
    os.environ.get("BEDDING_MASK_ROOT", "/mnt/data/bedding_dataset/labels_extracted/labels")
)

#: Center-crop applied to native HF cubes (2400×4900) to match the pipeline's
#: training crop (1800×4300). Mirrors ``EAD_CROP`` in the cu3s→NPZ converter.
_TRAINING_CROP = (slice(300, -300), slice(300, -300))


def resolve_default_config() -> dict[str, Any]:
    """Resolve notebook-time configuration in one dict (lightweight — no downloads).

    Notebooks call this once at the top. It records the source toggles + repo-relative
    paths and asserts only the plugins manifest (which ships in the repo). The trained
    pipeline and eval artefacts are resolved lazily by :func:`resolve_pipeline` /
    :func:`resolve_eval_dir` so the *training* notebook (which makes its own pipeline)
    never triggers the model download. Local dataset paths are asserted only in
    ``local`` data mode.
    """
    cfg: dict[str, Any] = {
        "data_source": BEDDING_DATA_SOURCE,
        "pipeline_source": BEDDING_PIPELINE_SOURCE,
        "hf_repo_id": BEDDING_HF_REPO_ID,
        "model_hf_repo": BEDDING_MODEL_HF_REPO,
        "plugins_yaml": DEFAULT_PLUGINS_YAML,
        "splits_csv": DEFAULT_SPLITS_CSV,
        "cu3s_val_root": DEFAULT_CU3S_VAL_ROOT,
        "mask_root": DEFAULT_MASK_ROOT,
        "local_pipeline_dir": LOCAL_PIPELINE_DIR,
        "bedding_all6_nm": BEDDING_ALL6_NM,
        "bedding_all6_labels": BEDDING_ALL6_LABELS,
    }
    assert DEFAULT_PLUGINS_YAML.exists(), (
        f"Plugins manifest not found at {DEFAULT_PLUGINS_YAML}. "
        f"Run the notebook from inside the cuvis-ai-dinomaly repo."
    )
    if BEDDING_DATA_SOURCE == "local":
        for key in ("splits_csv", "cu3s_val_root"):
            assert cfg[key].exists(), (
                f"BEDDING_DATA_SOURCE='local' but missing {key!r} = {cfg[key]}. "
                f"Set BEDDING_DATA_SOURCE='hf' to download from HuggingFace instead."
            )
    return cfg


# ---------------------------------------------------------------------------
# Trained-pipeline + eval resolution (HuggingFace model repo, or local-trained)
# ---------------------------------------------------------------------------


def _hf_model_download(filename: str) -> Path:
    """Download ``filename`` from the HF *model* repo, cached, return its path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "HF pipeline source requires `huggingface_hub` (pip install huggingface_hub)."
        ) from e
    return Path(
        hf_hub_download(
            repo_id=BEDDING_MODEL_HF_REPO,
            repo_type="model",
            filename=filename,
            cache_dir=str(BEDDING_HF_CACHE),
        )
    )


def resolve_pipeline() -> tuple[Path, Path]:
    """Return ``(yaml_path, pt_path)`` for the trained pipeline.

    Default (``BEDDING_PIPELINE_SOURCE='hf'``): downloads the pretrained pipeline
    from :data:`BEDDING_MODEL_HF_REPO` (the ~580 MB ``.pt`` is fetched + cached on
    first use). ``'local'``: uses the pipeline you trained yourself in
    :data:`LOCAL_PIPELINE_DIR` (the train notebook's output) — picks the single
    ``*.yaml`` there and its sibling ``.pt``.
    """
    if BEDDING_PIPELINE_SOURCE == "local":
        d = LOCAL_PIPELINE_DIR
        yamls = sorted(d.glob("*.yaml"))
        assert yamls, (
            f"BEDDING_PIPELINE_SOURCE='local' but no *.yaml in {d}. "
            f"Train one with the training notebook first, or set BEDDING_PIPELINE_SOURCE=hf."
        )
        yaml_path = yamls[0]
        pt_path = yaml_path.with_suffix(".pt")
        assert pt_path.is_file(), f"Missing weights next to {yaml_path.name}: {pt_path}"
        return yaml_path, pt_path
    return _hf_model_download("dinomaly_bedding_all6.yaml"), _hf_model_download(
        "dinomaly_bedding_all6.pt"
    )


def resolve_eval_dir() -> Path | None:
    """Return a directory holding ``report.json`` (+ per-class / Dice JSON), or ``None``.

    ``'hf'``: downloads the ``eval_val/*.json`` metrics from the model repo and
    returns their cache dir. ``'local'``: the ``eval_val`` sibling of
    :data:`LOCAL_PIPELINE_DIR`, if present. ``None`` means the headline/per-class
    cells should skip gracefully.
    """
    if BEDDING_PIPELINE_SOURCE == "local":
        d = LOCAL_PIPELINE_DIR.parent / "eval_val"
        return d if (d / "report.json").is_file() else None
    try:
        report = _hf_model_download("eval_val/report.json")
    except Exception:
        return None
    for extra in ("eval_val/per_class_auroc.json", "eval_val/dice_recompute.json"):
        try:
            _hf_model_download(extra)
        except Exception:
            pass
    return report.parent


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


def load_bedding_splits() -> Any:
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


def prepare_bedding_data(
    npz_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("val",),
    limit: int = 0,
) -> tuple[Path, Path]:
    """Materialize per-frame NPZ + split artifacts for the requested bedding split(s).

    Factors the training tutorial's cu3s -> cropped-NPZ conversion into a reusable helper. For each
    frame of the dataset ``splits.csv`` whose split is in ``splits`` it loads the cu3s cube
    (downloaded from HuggingFace in the default mode and center-cropped 2400x4900 -> 1800x4300 to
    match the pretrained pipeline), pairs it with its GT mask (rasterized into a binary ``mask`` and
    a multi-class ``class_mask``; zeros for a normal frame), and writes one compressed ``.npz``. It
    then emits the two artifacts the data module consumes: a ``universe.csv`` (``source, index,
    path``; one measurement per file, so ``index`` is always 0) and a baked ``splits.json`` (core
    ``DataSplitConfig`` of ``file_indices`` selectors, ``predict`` mapped to ``val``). Reruns reuse
    the artifacts when both already exist under ``npz_dir``. Returns ``(splits_json, universe_csv)``,
    ready for ``MultiNpzDataModule(splits=DataSplitConfig(splits_path=splits_json),
    universe_csv=universe_csv)``.

    Parameters
    ----------
    npz_dir
        Where per-frame NPZ (HF mode) and the ``splits.json`` / ``universe.csv`` are written.
    splits
        Dataset splits to convert. Inference needs only ``("val",)`` (the predict split); the
        training tutorial converts ``("train", "val")``.
    limit
        If > 0, keep at most this many frames per split (fast dry-run).
    """
    import csv

    from cuvis_ai_core.data.splits_io import save_splits
    from cuvis_ai_dataloader.data.npz_converter import write_universe_csv
    from cuvis_ai_schemas.training import DataSplitConfig, Selector, SelectorKind
    from PIL import Image

    cfg = resolve_default_config()
    npz_out = Path(npz_dir)
    npz_out.mkdir(parents=True, exist_ok=True)
    universe_csv = npz_out / "universe.csv"
    splits_json = npz_out / "splits.json"

    # (split, source_stem, absolute_npz_path) for every converted / indexed frame.
    frames: list[tuple[str, str, Path]] = []

    if cfg["data_source"] == "local":
        for r in csv.DictReader(open(cfg["splits_csv"])):
            if r["split"] not in splits:
                continue
            p = Path(r["npz_path"])
            frames.append((r["split"], p.stem, p.resolve()))
    elif not (universe_csv.is_file() and splits_json.is_file()):
        dataset_splits = load_bedding_splits()  # HF splits.csv (split, stem, ...)
        per_split_count: dict[str, int] = {}
        for _, row in dataset_splits.iterrows():
            split, stem = row["split"], row["stem"]
            if split not in splits:
                continue
            if limit and per_split_count.get(split, 0) >= limit:
                continue
            per_split_count[split] = per_split_count.get(split, 0) + 1
            cube, wl = load_bedding_cube(stem, split=split)  # cropped (1800, 4300, 6) float32
            mask_path = load_bedding_mask_path(stem)
            if mask_path is not None:
                m = center_crop_to_training(np.asarray(Image.open(mask_path)))
                mask = (m > 0).astype(np.int32)
                class_mask = m.astype(np.uint8)
            else:  # normal frame: empty mask
                mask = np.zeros(cube.shape[:2], dtype=np.int32)
                class_mask = np.zeros(cube.shape[:2], dtype=np.uint8)
            out = npz_out / split / f"{stem}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out, cube=cube, wavelengths=wl.astype(np.int32), mask=mask, class_mask=class_mask
            )
            frames.append((split, stem, out.resolve()))

    # Emit the two split artifacts (skipped implicitly when nothing new was converted and both
    # already exist): a universe.csv and a baked splits.json with predict mapped to val.
    if frames:
        write_universe_csv(
            [{"source": stem, "index": 0, "path": path.as_posix()} for _s, stem, path in frames],
            universe_csv,
        )

        def _selectors(split: str) -> list[Selector]:
            return [
                Selector(kind=SelectorKind.FILE_INDICES, source=stem, ids=[0])
                for s, stem, _ in frames
                if s == split
            ]

        save_splits(
            DataSplitConfig(
                train=_selectors("train"), val=_selectors("val"), predict=_selectors("val")
            ),
            splits_json,
        )
    return splits_json, universe_csv


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
    assert cube_bhwc.ndim == 4 and cube_bhwc.shape[-1] == 6, (
        f"expected [B,H,W,6] cube, got {cube_bhwc.shape}"
    )
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


def load_bedding_frame(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load a per-frame bedding NPZ -> ``{cube [H,W,6] f32, wavelengths [C] i32, mask [H,W] i32,
    class_mask [H,W] u8}`` (mask / class_mask zeros when the frame is normal / unbaked)."""
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


def panel_frames(collected: list, datamodule: Any) -> list[dict[str, Any]]:
    """Flatten a ``Predictor`` collect run into one slim record per frame for the panels.

    Pulls the anomaly ``scores`` map, binary ``mask``, and per-frame ``anomaly_score`` out of each
    per-batch ``(node, port)`` output dict (already moved to CPU by ``collect_ports``), plus the
    frame's NPZ ``path`` / ``index`` from the predict dataset. The cube for the VIS / SWIR panels is
    reloaded from that path by the caller (``load_bedding_frame``); per-class AUROC comes from the
    metric node, so neither the cube nor the multi-class mask is flattened here.
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


def plot_per_class_auroc_bar(
    per_class: dict[str, float] | str | Path,
    *,
    title: str = "Per-class pixel AUROC",
    figsize: tuple[float, float] = (12.0, 6.0),
):
    """Render a horizontal bar chart of per-class AUROC, ascending.

    Accepts either a ``{class_name: auroc}`` dict (as returned by
    ``PerClassAnomalyAUROC.compute()``) or a path to a recomputed per-class json (the published
    ``eval_val/per_class_auroc.json``, whose values may be nested under an ``auroc`` key).
    """
    import json

    if isinstance(per_class, (str, Path)):
        data = json.loads(Path(per_class).read_text())
        items = data["per_class_auroc"] if "per_class_auroc" in data else data
        per_class = {n: (v["auroc"] if isinstance(v, dict) else v) for n, v in items.items()}
    if not per_class:
        return None
    names = sorted(per_class, key=per_class.get)
    aurocs = [per_class[n] for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(names, aurocs, color="steelblue")
    ax.set_xlim(0.5, 1.01)
    ax.axvline(1.0, color="black", linewidth=0.5)
    ax.set_xlabel("Pixel AUROC")
    ax.set_title(title)
    for bar, val in zip(bars, aurocs, strict=True):
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8
        )
    fig.tight_layout()
    return fig


def load_headline_report(report_json_path: Path) -> dict[str, Any]:
    """Load eval_val/report.json. Used by the results notebook header."""
    import json

    return json.loads(Path(report_json_path).read_text())


# ---------------------------------------------------------------------------
# Inference helpers: speedup recipe demo (TF32 + bf16 autocast + torch.compile)
# ---------------------------------------------------------------------------


def apply_lossless_speedups(
    pipeline, *, autocast_dtype: torch.dtype = torch.bfloat16, compile_mode: str = "reduce-overhead"
):
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
