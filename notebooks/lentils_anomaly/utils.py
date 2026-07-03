"""Shared helpers for the lentils × Dinomaly tutorial notebooks.

Four notebooks ride on these helpers, all on the current **`dinomaly`** plugin
(`cuvis_ai_dinomaly` / `DinomalyDetector`) — never the defunct `dinomaly2`:

- ``lentils_rgb_train_tutorial.ipynb``          — train with the RGB fixed-wavelength selector
- ``lentils_cir_train_tutorial.ipynb``          — train with the CIR (NIR/Red/Green) selector
- ``lentils_adaclip_bands_train_tutorial.ipynb``— train with the 3 bands AdaCLIP's frozen concrete
                                                  selector converged to (indices 14/59/57), fixed
- ``lentils_inference_tutorial.ipynb``          — load a trained pipeline, eval on the 180 test,
                                                  per-class AUROC breakdown

Lentils dataset
---------------
61-channel VNIR (430–910 nm), foreign-object anomaly detection. Published on HuggingFace at
``cubert-gmbh/XMR_Industrial_Foreign_Object_Detection_Lentils`` (merged cu3s sessions per day +
per-day global COCO). The Dinomaly split (train-on-normals) is the HF ``splits_dinomaly.csv``:
train 308 (normal) / val 148 / test 180 / adaclip_train 500 (held out).

Workflow
--------
1. Download cu3s from HF (``huggingface-cli download`` / ``snapshot_download``).
2. Convert to per-frame NPZ with baked ``mask`` + ``class_mask`` via cuvis-ai-dataloader's
   ``cu3s-to-npz`` (NPZ are never stored on HF).
3. Train / infer from the NPZ via ``MultiNpzDataModule`` (``npz_multi``).

``LENTILS_DATA_SOURCE`` selects the data path: ``hf`` (default) downloads + converts on demand;
``local`` reads pre-existing NPZ from the CSV in ``LENTILS_SPLITS_CSV``.
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
LENTILS_HF_CACHE = Path(os.environ.get("LENTILS_HF_CACHE", str(Path.home() / ".cache" / "cuvis_lentils")))

#: "hf" (default) downloads cu3s from HF + converts to NPZ on demand; "local" reads pre-existing
#: NPZ from the CSV in ``LENTILS_SPLITS_CSV``. Env-overridable.
LENTILS_DATA_SOURCE = os.environ.get("LENTILS_DATA_SOURCE", "hf").lower()

#: The local (train-on-normals) split CSV of ``(split, npz_path, image_id)`` — only used in
#: ``local`` mode; set it via the ``LENTILS_SPLITS_CSV`` env var (no default path).
LOCAL_SPLITS_CSV = Path(os.environ["LENTILS_SPLITS_CSV"]) if os.environ.get("LENTILS_SPLITS_CSV") else None
#: Where HF cu3s are downloaded + converted to NPZ (hf mode).
LENTILS_NPZ_OUT = LENTILS_HF_CACHE / "npz"

#: Trained-pipeline dir the inference notebook reads. Defaults to the RGB train notebook's
#: output; set ``LENTILS_PIPELINE_DIR`` to evaluate a CIR / AdaCLIP-bands run instead.
LOCAL_PIPELINE_DIR = Path(
    os.environ.get(
        "LENTILS_PIPELINE_DIR",
        str(REPO_ROOT / "notebooks" / "lentils_anomaly" / "outputs" / "lentils_adaclip_run" / "trained_models"),
    )
)

#: COCO category id → name (per-day global COCO; 0 = background/unlabeled).
LENTILS_CATEGORIES: dict[int, str] = {
    0: "Unlabeled", 1: "stem_k", 2: "stone", 3: "alu_shard",
    4: "blue_paper", 5: "white_paper", 6: "fly", 7: "rubber",
}

#: The three cube-channel indices AdaCLIP's frozen concrete selector converged to on lentils.
#: Resolved to wavelengths per-dataset by :func:`resolve_adaclip_wavelengths` (typically
#: ~(542, 902, 886) nm on the 61-band lentils data), then fed to a FixedWavelengthSelector.
ADACLIP_BAND_INDICES: tuple[int, int, int] = (14, 59, 57)


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
        if LOCAL_SPLITS_CSV is None or not LOCAL_SPLITS_CSV.is_file():
            raise FileNotFoundError(
                "LENTILS_DATA_SOURCE='local' needs LENTILS_SPLITS_CSV pointing at a "
                f"(split, npz_path, image_id) CSV; got {LOCAL_SPLITS_CSV!r}. Use the default "
                "'hf' source to download + convert from HuggingFace instead."
            )
    return cfg


def resolve_pipeline() -> tuple[Path, Path]:
    """Return ``(yaml_path, pt_path)`` for a trained lentils pipeline (local).

    Picks the single ``*.yaml`` in :data:`LOCAL_PIPELINE_DIR` (a train notebook's
    ``trained_models`` dir) + its sibling ``.pt``. Set ``LENTILS_PIPELINE_DIR`` to choose a
    specific run (RGB / CIR / AdaCLIP-bands).
    """
    d = LOCAL_PIPELINE_DIR
    yamls = sorted(d.glob("*.yaml"))
    if not yamls:
        raise FileNotFoundError(
            f"No trained pipeline *.yaml in {d}. Run a train notebook first, or set "
            f"LENTILS_PIPELINE_DIR to a trained_models dir."
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
            nir_nm=860.0, red_nm=670.0, green_nm=560.0, norm_mode="running",
            running_warmup_frames=0, freeze_running_bounds_after_frames=20, name=name,
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
    splits_csv: str | Path, indices: tuple[int, int, int] = ADACLIP_BAND_INDICES
) -> tuple[float, float, float]:
    """Map AdaCLIP's frozen band ``indices`` to wavelengths (nm) using the first train NPZ.

    Mirrors ``examples/train_dinomaly_rgb_frozen_adaclip_bands_multifile.py``: reads the
    ``wavelengths`` array of the first ``split == train`` row (or the first row with an
    ``npz_path``) and returns ``(w[i0], w[i1], w[i2])`` in R,G,B order.
    """
    import csv as _csv

    with open(splits_csv, newline="") as f:
        rows = list(_csv.DictReader(f))
    train_rows = [r for r in rows if r.get("split") == "train" and (r.get("npz_path") or "").strip()]
    if not train_rows:
        train_rows = [r for r in rows if (r.get("npz_path") or "").strip()]
    if not train_rows:
        raise ValueError(f"No rows with npz_path in {splits_csv}")
    with np.load(train_rows[0]["npz_path"]) as z:
        if "wavelengths" not in z.files:
            raise KeyError(f"{train_rows[0]['npz_path']} has no 'wavelengths'")
        w = np.asarray(z["wavelengths"], dtype=np.float64).ravel()
    for i in indices:
        if not (0 <= i < len(w)):
            raise IndexError(f"band index {i} out of range for {len(w)} bands")
    return (float(w[indices[0]]), float(w[indices[1]]), float(w[indices[2]]))


# --------------------------------------------------------------------------- data
def resolve_splits_csv() -> Path:
    """Local ``(split, npz_path, image_id)`` split CSV (``local`` mode only).

    The ``hf`` path does not use this — it downloads + converts via :func:`ensure_lentils_npz`,
    which returns its own split CSV.
    """
    if LOCAL_SPLITS_CSV is None:
        raise FileNotFoundError(
            "LENTILS_DATA_SOURCE='local' needs LENTILS_SPLITS_CSV set to a "
            "(split, npz_path, image_id) CSV."
        )
    return LOCAL_SPLITS_CSV


def subsample_splits_csv(csv_path: str | Path, n_per_split: int, out_path: str | Path) -> Path:
    """Write a splits CSV keeping at most ``n_per_split`` rows per split (fast dry-runs).

    Rows whose ``split`` is not one of train/val/test are dropped (e.g. ``adaclip_train``,
    which the datamodule ignores anyway). Returns ``out_path``.
    """
    import csv as _csv
    from collections import defaultdict

    with open(csv_path, newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty splits CSV: {csv_path}")
    fields = list(rows[0].keys())
    kept: list[dict[str, str]] = []
    seen: dict[str, int] = defaultdict(int)
    for r in rows:
        s = r.get("split", "")
        if s not in ("train", "val", "test"):
            continue
        if seen[s] < n_per_split:
            kept.append(r)
            seen[s] += 1
    with open(out_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    return Path(out_path)


def ensure_lentils_npz(
    out_dir: str | Path, *, limit: int = 0, splits_csv: str | Path | None = None
) -> Path:
    """Download the lentils cu3s + per-session COCO from HF, convert to per-frame NPZ.

    Returns a splits CSV ``(split, npz_path, image_id)`` that ``MultiNpzDataModule`` reads.

    Each per-session ``.cu3s`` is indexed by its **measurement index = ``local_image_id``**
    (0..N-1), which is also the ``image_id`` in that session's sibling COCO (``json_path``). So we
    read + label frames by ``local_image_id`` (``frame_index == image_id`` — no decoupling needed;
    ``camera_frame_num`` is the original camera counter, not the cu3s index). Each cu3s is
    downloaded once and its needed frames converted together.

    Parameters
    ----------
    limit
        If > 0, keep at most this many frames per split (fast dry-run).
    splits_csv
        Use this pre-filtered splits CSV instead of downloading the HF ``splits_dinomaly.csv``
        (must carry ``cu3s_path, json_path, camera_frame_num, local_image_id, split``).
    """
    import csv as _csv
    from collections import OrderedDict, defaultdict

    from cuvis_ai_dataloader.data.npz_converter import convert_cu3s_file
    from huggingface_hub import hf_hub_download

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = str(LENTILS_HF_CACHE)

    if splits_csv is None:
        splits_csv = hf_hub_download(
            LENTILS_HF_REPO_ID, repo_type="dataset", filename="splits_dinomaly.csv", cache_dir=cache
        )
    with open(splits_csv, newline="") as f:
        rows = [r for r in _csv.DictReader(f) if r.get("split") in ("train", "val", "test")]

    if limit:
        seen: dict[str, int] = defaultdict(int)
        kept = []
        for r in rows:
            s = r["split"]
            if seen[s] < limit:
                kept.append(r)
                seen[s] += 1
        rows = kept

    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for r in rows:
        groups.setdefault(r["cu3s_path"], []).append(r)

    out_rows: list[dict[str, Any]] = []
    for cu3s_rel, grp in groups.items():
        json_rel = (grp[0].get("json_path") or "").strip()
        cu3s = hf_hub_download(LENTILS_HF_REPO_ID, repo_type="dataset", filename=cu3s_rel, cache_dir=cache)
        # No sibling COCO -> convert without masks (loader emits zeros; frames read as normal).
        coco = (
            hf_hub_download(LENTILS_HF_REPO_ID, repo_type="dataset", filename=json_rel, cache_dir=cache)
            if json_rel
            else None
        )
        recs = convert_cu3s_file(
            cu3s,
            out_dir,
            annotation_json=coco,
            frame_indices=[int(r["local_image_id"]) for r in grp],
        )
        for r, rec in zip(grp, recs, strict=True):
            out_rows.append({"split": r["split"], "npz_path": rec["npz_path"], "image_id": rec["image_id"]})

    csv_out = out_dir / "lentils_splits_npz_fromhf.csv"
    with open(csv_out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["split", "npz_path", "image_id"])
        w.writeheader()
        w.writerows(out_rows)
    return csv_out


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
    """One-vs-background pixel AUROC per non-background class (uses the baked ``class_mask``).

    Pools **raw** scores across frames — AUROC is rank-based, so per-frame min-max normalization
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


def run_test_inference(pipeline: Any, datamodule: Any, *, device: Any, limit: int = 0) -> list[dict[str, Any]]:
    """Run a loaded pipeline over the ``test`` split; return one result dict per frame.

    Each dict: ``score_map`` [H,W] f32, ``anomaly_score`` float|None, ``mask`` [H,W] i32,
    ``class_mask`` [H,W] u8, ``rgb`` [H,W,3] f32|None. Reuses the extraction helpers from
    ``examples/run_saved_dinomaly_pipeline_test_npz.py`` so the notebook path matches the CLI.
    The DINOv2 encoder runs under ``torch.no_grad()``.
    """
    import sys as _sys

    import torch as _torch
    from cuvis_ai_core.utils.graph_helper import restructure_output_to_node_dict
    from cuvis_ai_schemas.enums import ExecutionStage
    from cuvis_ai_schemas.execution import Context

    _ex = str(REPO_ROOT / "examples")
    if _ex not in _sys.path:
        _sys.path.insert(0, _ex)
    from run_saved_dinomaly_pipeline_test_npz import (
        _move_batch,
        _pick_dinomaly_outputs,
        _pick_selector_outputs,
        _sample_np,
    )

    loader = datamodule.test_dataloader()
    records = getattr(datamodule.test_ds, "records", None)
    if records is None:
        records = getattr(datamodule.test_ds, "_rows", None)
    results: list[dict[str, Any]] = []
    offset = 0
    for bidx, batch in enumerate(loader):
        if limit and offset >= limit:
            break
        batch = _move_batch(batch, device)
        bsz = int(batch["cube"].shape[0])
        ctx = Context(stage=ExecutionStage.TEST, epoch=0, batch_idx=bidx, global_step=offset)
        with _torch.no_grad():
            raw = pipeline.forward(batch=batch, context=ctx)
        node_out = restructure_output_to_node_dict(raw)
        dino = _pick_dinomaly_outputs(node_out)
        sel = _pick_selector_outputs(node_out)
        for i in range(bsz):
            if limit and offset + i >= limit:
                break
            score = _sample_np(dino.get("scores"), i, expected_batch_size=bsz)
            if score is not None and score.ndim == 3 and score.shape[-1] == 1:
                score = score[..., 0]
            ascore = _sample_np(dino.get("anomaly_score"), i, expected_batch_size=bsz)
            mask = _sample_np(batch.get("mask"), i, expected_batch_size=bsz)
            cmask = _sample_np(batch.get("class_mask"), i, expected_batch_size=bsz)
            rgb = _sample_np(sel.get("rgb_image"), i, expected_batch_size=bsz)
            rec = records[offset + i] if records is not None and offset + i < len(records) else {}
            results.append({
                "score_map": None if score is None else np.asarray(score, np.float32),
                "anomaly_score": None if ascore is None else float(np.asarray(ascore).ravel()[0]),
                "mask": None if mask is None else np.asarray(mask, np.int32),
                "class_mask": None if cmask is None else np.asarray(cmask, np.uint8),
                "rgb": None if rgb is None else np.asarray(rgb, np.float32),
                "npz_path": rec.get("npz_path") if isinstance(rec, dict) else None,
                "image_id": rec.get("image_id") if isinstance(rec, dict) else None,
            })
        offset += bsz
    return results


def overall_auroc(results: list[dict[str, Any]]) -> dict[str, float]:
    """Overall pixel + image AUROC over inference results (binary anomaly = ``mask != 0``).

    Pools **raw** scores across frames (AUROC is rank-based); per-frame normalization would not be
    a global monotonic transform and would distort the pooled pixel ranking. ``_norm`` is display-only.
    """
    from sklearn.metrics import roc_auc_score

    px_s, px_y, img_s, img_y = [], [], [], []
    for r in results:
        if r.get("score_map") is None or r.get("mask") is None:
            continue
        px_s.append(np.asarray(r["score_map"], np.float32).ravel())
        px_y.append((r["mask"].ravel() != 0).astype(int))
        img_y.append(int((r["mask"] != 0).any()))
        img_s.append(
            float(r["anomaly_score"]) if r.get("anomaly_score") is not None
            else float(np.asarray(r["score_map"]).max())
        )
    out: dict[str, float] = {}
    if px_y:
        py, ps = np.concatenate(px_y), np.concatenate(px_s)
        if 0 < int(py.sum()) < py.size:
            out["pixel_auroc"] = float(roc_auc_score(py, ps))
    if img_y:
        iy = np.asarray(img_y)
        if 0 < int(iy.sum()) < iy.size:
            out["image_auroc"] = float(roc_auc_score(iy, np.asarray(img_s)))
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
