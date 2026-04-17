"""Tests for NPZ dataset/datamodule paths used by plugin training examples."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from cuvis_ai_dinomaly.data import MultiFileNpzDataModule, MultiFileNpzDataset


def _write_npz(path: Path, *, with_mask: bool) -> None:
    h, w, c = 8, 10, 5
    cube = np.random.rand(h, w, c).astype(np.float32)
    wavelengths = np.linspace(450, 850, c).astype(np.float32)
    if with_mask:
        mask = np.zeros((h, w), dtype=np.int32)
        mask[2:5, 3:7] = 2
        np.savez(path, cube=cube, wavelengths=wavelengths, mask=mask)
    else:
        np.savez(path, cube=cube, wavelengths=wavelengths)


def _write_splits_csv(path: Path, npz_train: Path, npz_val: Path, npz_test: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["split", "npz_path", "cu3s_path", "annotation_json", "image_id"],
        )
        w.writeheader()
        w.writerow({"split": "train", "npz_path": str(npz_train), "cu3s_path": "a.cu3s", "annotation_json": "", "image_id": 1})
        w.writerow({"split": "val", "npz_path": str(npz_val), "cu3s_path": "b.cu3s", "annotation_json": "", "image_id": 2})
        w.writerow({"split": "test", "npz_path": str(npz_test), "cu3s_path": "c.cu3s", "annotation_json": "", "image_id": 3})


def test_npz_dataset_reads_mask_and_wavelengths(tmp_path: Path) -> None:
    npz = tmp_path / "frame.npz"
    _write_npz(npz, with_mask=True)
    ds = MultiFileNpzDataset([{"npz_path": str(npz), "annotation_json": "", "image_id": 7}])
    item = ds[0]
    assert item["cube"].shape == (8, 10, 5)
    assert item["mask"].shape == (8, 10)
    assert item["wavelengths"].shape == (5,)
    assert int(item["mesu_index"]) == 7
    assert ds.num_channels == 5


def test_npz_dataset_builds_empty_mask_when_not_in_npz(tmp_path: Path) -> None:
    npz = tmp_path / "frame_nomask.npz"
    _write_npz(npz, with_mask=False)
    ds = MultiFileNpzDataset([{"npz_path": str(npz), "annotation_json": "", "image_id": 11}])
    item = ds[0]
    assert item["mask"].shape == (8, 10)
    assert np.all(item["mask"] == 0)


def test_datamodule_setup_and_dataloaders(tmp_path: Path) -> None:
    npz_train = tmp_path / "train.npz"
    npz_val = tmp_path / "val.npz"
    npz_test = tmp_path / "test.npz"
    _write_npz(npz_train, with_mask=True)
    _write_npz(npz_val, with_mask=True)
    _write_npz(npz_test, with_mask=True)
    splits = tmp_path / "splits.csv"
    _write_splits_csv(splits, npz_train, npz_val, npz_test)

    dm = MultiFileNpzDataModule(splits_csv=splits, batch_size=2, num_workers=0)
    dm.setup("fit")
    assert dm.train_ds is not None
    assert dm.val_ds is not None
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 2

    dm.setup("test")
    assert dm.test_ds is not None
    test_loader = dm.test_dataloader()
    assert test_loader.batch_size == 2


def test_datamodule_raises_if_loader_called_before_setup(tmp_path: Path) -> None:
    splits = tmp_path / "empty.csv"
    with splits.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["split", "npz_path", "cu3s_path", "annotation_json", "image_id"],
        )
        w.writeheader()

    dm = MultiFileNpzDataModule(splits_csv=splits, batch_size=1, num_workers=0)
    with pytest.raises(RuntimeError):
        dm.train_dataloader()
    with pytest.raises(RuntimeError):
        dm.val_dataloader()
    with pytest.raises(RuntimeError):
        dm.test_dataloader()
