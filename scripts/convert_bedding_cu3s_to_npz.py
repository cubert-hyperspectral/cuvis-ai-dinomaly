"""Convert bedding cu3s frames → per-frame NPZ + splits CSV.

Conforms to `MultiFileNpzDataset` schema (cuvis_ai_dinomaly.data.multi_file_npz_dataset).

For each input cu3s:
  1. Load `(2400, 4900, 6)` uint16 cube via cuvis.SessionFile. The cu3s already
     carry the reflectance calibration; cuvis returns reflectance × 10000
     (uint16, max ~38000 with specular highlights). This matches lentils
     convention (max ~10000) so the downstream pipeline behaves consistently.
  2. Crop tray borders: `cube[300:-300, 300:-300, :]` → `(1800, 4300, 6)`.
  3. Cast to float32 only (NO divide-by-10000, NO 0.55 factor).
     Rationale (decision 2026-05-27): DinomalyDetector's
     `_rgb_bhwc_to_model_input` does per-cube max-scaling to [0, 1] before
     ImageNet normalisation. Pre-dividing by 10000 here would land us at
     values 0–~3.2 — the auto-scaler's old uint8 path would treat that as
     [0, 255] and crush everything to ~0. Keeping the raw u16-equivalent
     max (~38000) ensures the auto-scaler does `x / max_val` cleanly. EAD's
     `* 0.55 / 10000` step is independent of this change.
  4. **No clipping.** Specular highlights (~p99 = 16500 in u16 reflectance
     scale = 1.65 in raw reflectance) preserved.
  5. **No spatial resize** — keep full (1800, 4300, 6). The DinomalyDetector node
     applies Resize → CenterCrop → Normalize at training time.
  6. Find the binary mask: cube-side `<id>_mask.png` → fallback RGB-side
     `<id>_<id>_(0000|0)_RGB_mask.png`. Crop matching `[300:-300, 300:-300]`,
     binarise `>0 → 1`, store as `mask` (int32). Also store the multi-class
     uint8 as `class_mask` for per-class breakdown.
  7. Save `<frame_id>.npz` with keys: `cube, wavelengths, mask, class_mask, source_cu3s`.

Then writes `bedding_splits_npz.csv` with columns:
  `npz_path, mask_path, annotation_json, split`
Where `mask_path` and `annotation_json` are empty strings (mask lives inside the NPZ).

`split=train` → all train cubes. `split=val` → all val cubes (clean + anomalous;
the 8 unlabelled val frames get the NPZ but with no `mask` key).

Run:
  CUVIS=/lib/cuvis <venv>/bin/python convert_bedding_cu3s_to_npz.py \
      --in-root /mnt/data/bedding_dataset/exported \
      --labels-root /mnt/data/bedding_dataset/labels_extracted/labels \
      --out-root /mnt/data/bedding_dataset_npz
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cuvis
import numpy as np
from PIL import Image

EAD_CROP = (slice(300, -300), slice(300, -300), slice(None))
EXPECTED_WL = [450, 550, 625, 1050, 1200, 1450]


def find_mask_png(labels_dir: Path, frame_id: str) -> Path | None:
    """Prefer cube-side mask, fall back to RGB-side. Both are 2400×4900 uint8."""
    cube_side = labels_dir / f"{frame_id}_mask.png"
    if cube_side.exists():
        return cube_side
    for pat in (f"{frame_id}_{frame_id}_0000_RGB_mask.png",
                f"{frame_id}_{frame_id}_0_RGB_mask.png"):
        p = labels_dir / pat
        if p.exists():
            return p
    return None


def load_cube(cu3s_path: Path) -> tuple[np.ndarray, list[int]]:
    sf = cuvis.SessionFile(str(cu3s_path))
    mesu = sf[0]
    return mesu.data["cube"].array, list(mesu.data["cube"].wavelength)


def convert_one(cu3s: Path, labels_dir: Path, out_dir: Path,
                split: str) -> tuple[Path, bool, int, list[int]]:
    """Return (npz_path, has_mask, pos_pixel_count, classes_present)."""
    cube_u16, wl = load_cube(cu3s)
    if wl != EXPECTED_WL:
        raise ValueError(f"wavelength mismatch on {cu3s.name}: {wl}")
    if cube_u16.shape != (2400, 4900, 6):
        raise ValueError(f"shape mismatch on {cu3s.name}: {cube_u16.shape}")

    # Crop, cast, scale — NO 0.55, NO clip
    # Cast to float32; do NOT scale — see module docstring for rationale.
    cube_cropped = cube_u16[EAD_CROP].astype(np.float32)
    # Shape now (1800, 4300, 6)

    # Mask
    has_mask = False
    pos = 0
    classes: list[int] = []
    payload: dict[str, np.ndarray | str] = {
        "cube": cube_cropped,
        "wavelengths": np.array(EXPECTED_WL, dtype=np.int32),
        "source_cu3s": str(cu3s),
    }

    mp = find_mask_png(labels_dir, cu3s.stem)
    if mp is not None:
        mask_full = np.array(Image.open(mp), dtype=np.uint8)
        if mask_full.shape != (2400, 4900):
            raise ValueError(f"mask shape {mask_full.shape} != (2400,4900) on {cu3s.name}")
        mask_cropped = mask_full[300:-300, 300:-300]
        if (mask_cropped > 0).any():
            has_mask = True
            pos = int((mask_cropped > 0).sum())
            classes = sorted(int(x) for x in np.unique(mask_cropped) if x != 0)
            payload["mask"] = (mask_cropped > 0).astype(np.int32)
            payload["class_mask"] = mask_cropped.astype(np.uint8)

    out = out_dir / f"{cu3s.stem}.npz"
    np.savez_compressed(out, **payload)
    return out, has_mask, pos, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True, type=Path,
                    help="Path to exported/ (contains train/ and val/)")
    ap.add_argument("--labels-root", required=True, type=Path,
                    help="Path to labels_extracted/labels/")
    ap.add_argument("--out-root", required=True, type=Path,
                    help="Output dir for NPZ + splits CSV")
    ap.add_argument("--limit", type=int, default=0,
                    help="Convert at most N files (per split) for smoke-testing")
    ap.add_argument("--splits-csv", type=str, default="bedding_splits_npz.csv")
    args = ap.parse_args()

    train_dir = args.in_root / "train"
    val_dir = args.in_root / "val"
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "train").mkdir(exist_ok=True)
    (args.out_root / "val").mkdir(exist_ok=True)

    rows: list[dict[str, str]] = []
    summary = {"train": {"n": 0, "with_mask": 0}, "val": {"n": 0, "with_mask": 0}}

    for split, src_dir in [("train", train_dir), ("val", val_dir)]:
        out_dir = args.out_root / split
        files = sorted(src_dir.glob("*.cu3s"))
        if args.limit:
            files = files[: args.limit]
        print(f"\n=== {split} ({len(files)} cu3s → {out_dir}) ===")
        for i, p in enumerate(files, 1):
            try:
                npz, has_mask, pos, classes = convert_one(p, args.labels_root, out_dir, split)
            except Exception as e:
                print(f"  [{i:>3}/{len(files)}] FAIL {p.name}: {e}", file=sys.stderr)
                continue
            summary[split]["n"] += 1
            summary[split]["with_mask"] += int(has_mask)
            rows.append({
                "npz_path": str(npz),
                "cu3s_path": "",          # MultiFileNpzDataset uses npz_path only
                "annotation_json": "",    # COCO JSONs ignored (50/53 empty on this dataset)
                "image_id": 0,            # one frame per cu3s → constant 0; needed by MultiFileNpzDataset
                "mask_path": "",          # mask lives inside the NPZ
                "split": split,
            })
            cls_s = f" classes={classes}" if classes else ""
            print(f"  [{i:>3}/{len(files)}] {p.stem[-30:]:<30s}  mask={has_mask}  pos={pos:>8d}{cls_s}")

    csv_path = args.out_root / args.splits_csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["npz_path", "cu3s_path", "annotation_json", "image_id", "mask_path", "split"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {csv_path}")
    for s in ("train", "val"):
        print(f"  {s}: {summary[s]['n']} cubes, {summary[s]['with_mask']} with mask")


if __name__ == "__main__":
    main()
