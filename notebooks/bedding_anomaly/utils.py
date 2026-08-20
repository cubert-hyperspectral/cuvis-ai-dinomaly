"""Thin matplotlib rendering helpers for the bedding Dinomaly tutorial notebooks.

Two notebooks ride on these helpers:

- ``bedding_all6_train_tutorial.ipynb``: build + train + save the 6-channel pipeline
- ``bedding_all6_inference_tutorial.ipynb``: load + run + per-class AUROC breakdown

Everything substantive (data provisioning, pipeline wiring, training, metrics) lives inline
in the notebook cells; this module only renders what the pipeline produced. The bedding
dataset is a 6-channel hyperspectral still-image set (450 / 550 / 625 nm VIS plus
1050 / 1200 / 1450 nm SWIR), so the panels here show a VIS triplet and a SWIR pseudo-RGB
triplet side by side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

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


def normalize_for_display(x: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1] for matplotlib imshow.

    Robust to all-zero arrays (returns zeros) and clips floats.
    """
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def split_cube_vis_swir(cube_bhwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 6-channel cube ``[B, H, W, 6]`` into VIS (R/G/B at 625/550/450 nm)
    and SWIR-as-pseudo-RGB (R/G/B at 1450/1200/1050 nm) views.

    The input channel order is the model's (625, 550, 450, 1450, 1200, 1050) nm layout,
    long wavelength first within each triplet, so the displayed VIS/SWIR triplets honour
    the descending-wavelength-to-R/G/B convention.

    Both outputs are returned in ``[H, W, 3]`` shape (batch dim squeezed for
    visualisation; if B>1 only the first item is used).
    """
    assert cube_bhwc.ndim == 4 and cube_bhwc.shape[-1] == 6, (
        f"expected [B,H,W,6] cube, got {cube_bhwc.shape}"
    )
    vis = cube_bhwc[0, ..., :3]
    swir = cube_bhwc[0, ..., 3:]
    return vis, swir


def render_inference_panel(
    cube_bhwc: np.ndarray,
    score_map: np.ndarray,
    *,
    gt_mask: np.ndarray | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (16.0, 4.0),
) -> Any:
    """Render the per-frame qualitative story: VIS, SWIR, score heatmap, GT overlay.

    Parameters
    ----------
    cube_bhwc : np.ndarray
        Input cube ``[B, H, W, 6]`` (batch=1 OK; first item is used).
    score_map : np.ndarray
        Pixel-wise anomaly score, ``[H, W]`` or ``[1, H, W, 1]``.
    gt_mask : np.ndarray, optional
        Binary ground-truth mask, same H/W as the score map.
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
                    "path": rec.get("materialized_path") if isinstance(rec, dict) else None,
                    "index": rec.get("index") if isinstance(rec, dict) else None,
                    "is_anomalous": bool(m is not None and m.any()),
                }
            )
        offset += bsz
    return frames


def plot_per_class_auroc_bar(
    per_class: dict[str, float],
    *,
    title: str = "Per-class pixel AUROC",
    figsize: tuple[float, float] = (12.0, 6.0),
) -> Any:
    """Render a horizontal bar chart of per-class AUROC, ascending.

    Takes the ``{class_name: auroc}`` dict returned by ``PerClassAnomalyAUROC.compute()``.
    """
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
