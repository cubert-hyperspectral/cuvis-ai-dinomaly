"""NPZ dataset/datamodule compatible with the CU3S node contract."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
from loguru import logger
from skimage.draw import polygon2mask
from torch.utils.data import DataLoader, Dataset

from cuvis_ai_dinomaly.data._coco_utils import _build_category_mask, _parse_coco_json


def _parse_coco_json(json_path: Path) -> dict[str, Any]:
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)
    cat_map = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
    return {"anns_by_image": dict(anns_by_image), "cat_map": cat_map}


def _build_category_mask(anns: list[dict[str, Any]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.int32)
    for ann in anns:
        cat_id = int(ann.get("category_id", 0))
        segs = ann.get("segmentation", [])
        if not isinstance(segs, list):
            continue
        for seg in segs:
            if not isinstance(seg, list) or len(seg) < 6:
                continue
            xy = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
            try:
                poly = polygon2mask((h, w), xy[:, [1, 0]])
                mask[poly] = cat_id
            except Exception:
                continue
    return mask


class MultiFileNpzDataset(Dataset):
    """Dataset for one-frame-per-file compressed NPZ records.

    Expected arrays in each NPZ:
    - cube: [H, W, C] float32
    - wavelengths: [C] int/float (cast to int32 for node compatibility)
    Optional:
    - mask: [H, W] int32
    """

    def __init__(self, frame_records: list[dict[str, Any]]) -> None:
        self.records = frame_records

        self._ann_cache: dict[str, dict[str, Any]] = {}
        for rec in self.records:
            jp = str(rec.get("annotation_json", ""))
            if jp and jp not in self._ann_cache and Path(jp).is_file():
                self._ann_cache[jp] = _parse_coco_json(Path(jp))

        if self.records:
            with np.load(self.records[0]["npz_path"]) as z:
                wl = np.asarray(z["wavelengths"]).ravel()
            self.wavelengths_nm = wl.astype(np.int32, copy=False)
            self.num_channels = int(self.wavelengths_nm.shape[0])
        else:
            self.wavelengths_nm = np.array([], dtype=np.int32)
            self.num_channels = 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        rec = self.records[idx]
        npz_path = Path(rec["npz_path"])
        image_id = int(rec["image_id"])

        with np.load(npz_path) as z:
            cube = np.asarray(z["cube"], dtype=np.float32)
            wavelengths = np.asarray(z["wavelengths"]).ravel().astype(np.int32, copy=False)
            mask_from_npz = np.asarray(z["mask"], dtype=np.int32) if "mask" in z.files else None
            # Multi-class mask (uint8) preserved by convert_bedding_cu3s_to_npz.py for
            # per-class evaluation. Absent on the all-background val frames.
            class_mask_from_npz = (
                np.asarray(z["class_mask"], dtype=np.uint8) if "class_mask" in z.files else None
            )

        mask: np.ndarray
        if mask_from_npz is not None:
            mask = mask_from_npz
        else:
            jp = str(rec.get("annotation_json", ""))
            if jp and jp in self._ann_cache:
                anns = self._ann_cache[jp]["anns_by_image"].get(image_id, [])
            else:
                anns = []
            mask = _build_category_mask(anns, cube.shape[0], cube.shape[1])

        if class_mask_from_npz is None:
            # Frame has no annotated anomalies (all-background) — emit zeros so collate
            # shapes line up; per-class AUROC will simply see no positives here.
            class_mask_from_npz = np.zeros((cube.shape[0], cube.shape[1]), dtype=np.uint8)

        return {
            "cube": cube,
            "mask": mask,
            "class_mask": class_mask_from_npz,
            "wavelengths": wavelengths,
            "mesu_index": image_id,
        }


class MultiFileNpzDataModule(pl.LightningDataModule):
    """DataModule for split CSVs containing NPZ frame records."""

    def __init__(
        self,
        splits_csv: str | Path,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        worker_multiprocessing_context: str = "spawn",
    ) -> None:
        super().__init__()
        self.splits_csv = Path(splits_csv)
        self.batch_size = int(batch_size)
        self.num_workers = max(0, int(num_workers))
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers and self.num_workers > 0)
        self.worker_multiprocessing_context = worker_multiprocessing_context
        self.train_ds: MultiFileNpzDataset | None = None
        self.val_ds: MultiFileNpzDataset | None = None
        self.test_ds: MultiFileNpzDataset | None = None

    def _load_records(self) -> dict[str, list[dict[str, Any]]]:
        records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        with self.splits_csv.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row.get("split", "")
                if split not in records:
                    continue
                npz_path = row.get("npz_path", "")
                if not npz_path:
                    continue
                records[split].append(
                    {
                        "npz_path": npz_path,
                        "cu3s_path": row.get("cu3s_path", ""),
                        "annotation_json": row.get("annotation_json", ""),
                        "image_id": int(row["image_id"]),
                    }
                )
        return records

    def setup(self, stage: str | None = None) -> None:
        records = self._load_records()
        if stage == "fit" or stage is None:
            if records["train"]:
                self.train_ds = MultiFileNpzDataset(records["train"])
                logger.info(f"NPZ train dataset: {len(self.train_ds)} frames")
            if records["val"]:
                self.val_ds = MultiFileNpzDataset(records["val"])
                logger.info(f"NPZ val dataset: {len(self.val_ds)} frames")
        if stage == "test" or stage is None:
            if records["test"]:
                self.test_ds = MultiFileNpzDataset(records["test"])
                logger.info(f"NPZ test dataset: {len(self.test_ds)} frames")

    def _loader(self, ds: Dataset, *, shuffle: bool) -> DataLoader:
        kwargs: dict[str, Any] = {
            "dataset": ds,
            "shuffle": shuffle,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if self.num_workers > 0 and self.worker_multiprocessing_context:
            kwargs["multiprocessing_context"] = self.worker_multiprocessing_context
        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("Train dataset not initialized. Call setup('fit').")
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("Val dataset not initialized. Call setup('fit').")
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("Test dataset not initialized. Call setup('test').")
        return self._loader(self.test_ds, shuffle=False)
