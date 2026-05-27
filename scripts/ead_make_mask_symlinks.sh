#!/usr/bin/env bash
# Create the symlinks EfficientAD's reporting.py needs to find PNG masks.
#
# `EfficientADCuvisDataSet` looks for `<cube_stem>_0_RGB_mask.png` *next to* the
# .cu3s file. Our bedding masks live in `labels_extracted/labels/` under
# `<cube_stem>_<cube_stem>_(0000|0)_RGB_mask.png` (and a cube-side variant for
# 1/59 frames). This script links them into `exported/val/` so EAD's loader
# picks them up, without copying ~3 GB of PNGs.
#
# Usage:
#   bash ead_make_mask_symlinks.sh \
#     /mnt/data/bedding_dataset/exported/val \
#     /mnt/data/bedding_dataset/labels_extracted/labels
set -euo pipefail
VAL_DIR="${1:-/mnt/data/bedding_dataset/exported/val}"
LABELS_DIR="${2:-/mnt/data/bedding_dataset/labels_extracted/labels}"

[[ -d "$VAL_DIR"    ]] || { echo "val dir not found: $VAL_DIR";    exit 1; }
[[ -d "$LABELS_DIR" ]] || { echo "labels dir not found: $LABELS_DIR"; exit 1; }

linked=0
missing=0
for cube in "$VAL_DIR"/*.cu3s; do
  stem="$(basename "$cube" .cu3s)"
  link="$VAL_DIR/${stem}_0_RGB_mask.png"
  [[ -e "$link" || -L "$link" ]] && continue

  # priority: cube-side, then RGB-side with two index conventions
  for cand in \
      "$LABELS_DIR/${stem}_mask.png" \
      "$LABELS_DIR/${stem}_${stem}_0000_RGB_mask.png" \
      "$LABELS_DIR/${stem}_${stem}_0_RGB_mask.png"; do
    if [[ -f "$cand" ]]; then
      ln -s "$cand" "$link"
      linked=$((linked+1))
      break
    fi
  done
  [[ -e "$link" ]] || { echo "  no mask for $stem"; missing=$((missing+1)); }
done

echo "linked=$linked missing=$missing"
