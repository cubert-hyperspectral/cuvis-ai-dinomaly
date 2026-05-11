"""Minimal COCO JSON utilities.

Inlined so the plugin does not depend on the unreleased ``cuvis_ai.data``
private helpers. Only polygon + bbox segmentation is supported (covers all
standard COCO use cases).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _parse_coco_json(path: Path) -> dict[str, Any]:
    """Parse a COCO-format JSON annotation file.

    Returns a dict with ``anns_by_image``: a mapping of ``image_id`` (int)
    to the list of annotation dicts for that image.
    """
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        iid = int(ann["image_id"])
        anns_by_image.setdefault(iid, []).append(ann)
    return {"anns_by_image": anns_by_image}


def _build_category_mask(
    anns: list[dict[str, Any]], height: int, width: int
) -> np.ndarray:
    """Build a ``[H, W]`` int32 category mask from COCO annotations.

    Polygon segmentation takes priority; falls back to bbox when no
    segmentation polygon is present. Each pixel is assigned the
    ``category_id`` of the covering annotation (last writer wins for
    overlapping regions). Returns an all-zeros mask when ``anns`` is empty.
    """
    mask = np.zeros((height, width), dtype=np.int32)
    if not anns:
        return mask

    import cv2  # opencv-python is a declared runtime dep

    for ann in anns:
        cat_id = int(ann.get("category_id", 1))
        seg = ann.get("segmentation", [])
        if seg and isinstance(seg, list) and not isinstance(seg[0], dict):
            # Standard polygon segmentation: list of [x0,y0,x1,y1,...] lists
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                pts_int = np.round(pts).astype(np.int32)
                cv2.fillPoly(mask, [pts_int], cat_id)
        else:
            # Fallback: bounding box [x, y, w, h] in COCO pixel coords
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                x, y, bw, bh = bbox
                x1 = max(0, int(round(x)))
                y1 = max(0, int(round(y)))
                x2 = min(width, int(round(x + bw)))
                y2 = min(height, int(round(y + bh)))
                mask[y1:y2, x1:x2] = cat_id
    return mask
