"""The lentils examples' data module, built from a trainrun's ``data`` section."""

from __future__ import annotations

from typing import Any

from cuvis_ai_core.data.splits_io import load_splits


def build_cu3s_datamodule(data: Any) -> Any:
    """Build the cu3s data module the examples read from and set it up for fitting.

    ``data.universe_csv`` lists the frames; ``data.splits_json`` (a core ``DataSplitConfig``)
    assigns them to train, val and test, and without it the universe.csv needs a ``split``
    column. ``data.reader`` holds the ``MultiCu3sDataModule`` reader options and is passed
    through as it is (``max_open_sessions``, ``read_ahead``, ``cuda_cubes``, ``sdk_cuda``,
    ``read_threads``).
    """
    # The examples extra installs the dataloader; the dev environment does not.
    from cuvis_ai_dataloader.data import MultiCu3sDataModule

    splits_json = data.get("splits_json")
    datamodule = MultiCu3sDataModule(
        universe_csv=str(data.universe_csv),
        splits=load_splits(splits_json) if splits_json else None,
        batch_size=int(data.batch_size),
        num_workers=int(data.get("num_workers", 0)),
        processing_mode=str(data.processing_mode),
        **(data.get("reader") or {}),
    )
    datamodule.setup(stage="fit")
    return datamodule
