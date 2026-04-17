"""Train Dinomaly on multi-file data with RGB from frozen AdaCLIP selector bands.

This script is intentionally standalone for one use case:
- Use AdaCLIP frozen concrete-selector winning channel indices [14, 59, 57]
- Resolve those indices to wavelengths from the first train NPZ in splits CSV
- Build Dinomaly pipeline with FixedWavelengthSelector(target_wavelengths=(lambda14, lambda59, lambda57))
- Train for 50 epochs by default (from Hydra config)

Expected resolved wavelengths on lentils 61-band data are typically:
(542.0, 902.0, 886.0) in R,G,B order.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from cuvis_ai.data import MultiFileCu3sDataModule
from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node.channel_selector import FixedWavelengthSelector
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.pipeline import PipelineMetadata
from cuvis_ai_schemas.training import CallbacksConfig, ModelCheckpointConfig, TrainingConfig
from cuvis_ai_core.training.config import create_callbacks_from_config
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cuvis_ai_dinomaly.data import MultiFileNpzDataModule

FROZEN_ADACLIP_CONCRETE_BAND_INDICES: tuple[int, int, int] = (14, 59, 57)


def _wavelengths_nm_for_band_indices(
    splits_csv: Path,
    indices: tuple[int, int, int],
) -> tuple[float, float, float]:
    with splits_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    train_rows = [r for r in rows if r.get("split") == "train" and (r.get("npz_path") or "").strip()]
    if not train_rows:
        train_rows = [r for r in rows if (r.get("npz_path") or "").strip()]
    if not train_rows:
        raise ValueError(f"No rows with npz_path found in {splits_csv}")

    npz_path = Path(train_rows[0]["npz_path"])
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    z = np.load(npz_path)
    if "wavelengths" not in z:
        raise KeyError(f"{npz_path} does not contain 'wavelengths'")
    w = np.asarray(z["wavelengths"], dtype=np.float64).ravel()

    vals: list[float] = []
    for idx in indices:
        if idx < 0 or idx >= len(w):
            raise IndexError(f"Band index {idx} out of range for {len(w)} bands")
        vals.append(float(w[idx]))
    return (vals[0], vals[1], vals[2])


@hydra.main(
    config_path="../configs",
    config_name="trainrun/dinomaly_multifile_rgb_frozen_adaclip_bands",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    logger.info("=== Dinomaly + MultiFile (RGB from frozen AdaCLIP concrete bands) ===")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Lightning checkpoint atomic save can use TMPDIR; keep scratch on the data disk.
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)

    plugins_manifest = Path(__file__).resolve().parent / "plugins.yaml"
    registry = NodeRegistry()
    registry.load_plugins(str(plugins_manifest))
    DinomalyDetector = NodeRegistry.get("cuvis_ai_dinomaly.node.dinomaly_detector.DinomalyDetector")
    DinomalyTrainLossBridge = NodeRegistry.get(
        "cuvis_ai_dinomaly.node.dinomaly_train_loss_bridge.DinomalyTrainLossBridge",
    )

    splits_csv = Path(cfg.data.splits_csv)
    target_wl = _wavelengths_nm_for_band_indices(splits_csv, FROZEN_ADACLIP_CONCRETE_BAND_INDICES)
    logger.info(
        "Using band indices {} -> target wavelengths (R,G,B) nm = {}",
        FROZEN_ADACLIP_CONCRETE_BAND_INDICES,
        target_wl,
    )
    (output_dir / "wavelength_selection.json").write_text(
        json.dumps(
            {
                "band_indices_rgb_order": list(FROZEN_ADACLIP_CONCRETE_BAND_INDICES),
                "target_wavelengths_nm_rgb_order": list(target_wl),
                "splits_csv": str(splits_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    backend = "cu3s"
    if splits_csv.is_file():
        header = splits_csv.open(encoding="utf-8").readline()
        if "npz_path" in header:
            backend = "npz"

    common_loader_kwargs = dict(
        splits_csv=cfg.data.splits_csv,
        batch_size=cfg.data.batch_size,
        num_workers=int(cfg.data.get("num_workers", 0)),
        pin_memory=bool(cfg.data.get("pin_memory", True)),
        persistent_workers=bool(cfg.data.get("persistent_workers", True)),
        worker_multiprocessing_context=str(cfg.data.get("worker_multiprocessing_context", "spawn")),
    )

    if backend == "npz":
        datamodule = MultiFileNpzDataModule(**common_loader_kwargs)
    else:
        datamodule = MultiFileCu3sDataModule(
            **common_loader_kwargs,
            processing_mode=cfg.data.processing_mode,
        )
    datamodule.setup(stage="fit")

    dcfg = cfg.dinomaly
    pipeline = CuvisPipeline("dinomaly_multifile_rgb_frozen_adaclip_bands")

    data_node = LentilsAnomalyDataNode(normal_class_ids=[0])
    normalizer = MinMaxNormalizer(
        eps=1e-6,
        use_running_stats=True,
        max_initialization_frames=cfg.get("minmax_init_frames", None),
    )
    selector = FixedWavelengthSelector(target_wavelengths=target_wl, name="rgb_selector")
    selector._requires_initial_fit_override = False

    dinomaly = DinomalyDetector(
        encoder_name=str(dcfg.encoder_name),
        bottleneck_dropout=float(dcfg.bottleneck_dropout),
        decoder_depth=int(dcfg.decoder_depth),
        image_size=int(dcfg.image_size),
        crop_size=int(dcfg.crop_size),
        use_center_crop=bool(dcfg.get("use_center_crop", False)),
        name="dinomaly_detector",
    )
    loss_bridge = DinomalyTrainLossBridge(weight=1.0, name="dinomaly_train_loss")
    decider = QuantileBinaryDecider(quantile=0.995, name="decider")
    metrics_node = AnomalyDetectionMetrics(name="metrics_anomaly")
    tb = TensorBoardMonitorNode(
        output_dir=str(output_dir / "tensorboard"),
        run_name=pipeline.name,
    )

    pipeline.connect(
        (data_node.outputs.cube, normalizer.data),
        (normalizer.normalized, selector.cube),
        (data_node.outputs.wavelengths, selector.wavelengths),
        (selector.rgb_image, dinomaly.rgb_image),
        (dinomaly.outputs.training_loss, loss_bridge.raw_loss),
        (dinomaly.outputs.scores, decider.logits),
        (dinomaly.outputs.scores, metrics_node.logits),
        (decider.decisions, metrics_node.decisions),
        (data_node.outputs.mask, metrics_node.targets),
        (metrics_node.metrics, tb.metrics),
    )

    pipeline.visualize(
        format="render_graphviz",
        output_path=str(output_dir / "pipeline" / f"{pipeline.name}.png"),
        show_execution_stage=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Moving pipeline to device: {}", device)
    pipeline.to(device)

    training_cfg = TrainingConfig.from_dict(OmegaConf.to_container(cfg.training, resolve=True))
    if training_cfg.trainer.callbacks is None:
        training_cfg.trainer.callbacks = CallbacksConfig()
    training_cfg.trainer.callbacks.checkpoint = ModelCheckpointConfig(
        dirpath=str(output_dir / "checkpoints"),
        monitor="metrics_anomaly/iou",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="{epoch:02d}",
        verbose=True,
    )

    if normalizer.requires_initial_fit:
        logger.info("Phase 1: Statistical initialization (MinMaxNormalizer)...")
        StatisticalTrainer(pipeline=pipeline, datamodule=datamodule).fit()

    unfreeze_names = list(cfg.unfreeze_nodes) if "unfreeze_nodes" in cfg else ["dinomaly_detector"]
    pipeline.unfreeze_nodes_by_name(unfreeze_names)
    logger.info("Unfrozen: {}", unfreeze_names)
    pipeline.to(device)

    logger.info("Gradient training...")
    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=[loss_bridge],
        metric_nodes=[metrics_node],
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
        monitors=[tb],
        callbacks=list(create_callbacks_from_config(training_cfg.trainer.callbacks)),
    )
    grad_trainer.fit()

    grad_trainer.datamodule.setup(stage="test")
    logger.info("Validation with best checkpoint...")
    grad_trainer.validate(ckpt_path="best")
    logger.info("Test with best checkpoint...")
    grad_trainer.test(ckpt_path="best")

    results_dir = output_dir / "trained_models"
    results_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = results_dir / f"{pipeline.name}.yaml"
    pipeline.save_to_file(
        str(pipeline_path),
        metadata=PipelineMetadata(
            name=pipeline.name,
            description=(
                "Dinomaly (Anomalib) on multi-file lentils, RGB selector from "
                f"frozen AdaCLIP concrete indices {FROZEN_ADACLIP_CONCRETE_BAND_INDICES} "
                f"(nm={target_wl})."
            ),
            tags=["dinomaly", "anomalib", "multifile", "rgb", "custom_wavelengths"],
            author="cuvis.ai",
        ),
    )

    logger.info("Pipeline saved: {}", pipeline_path)
    logger.info("TensorBoard: uv run tensorboard --logdir={}", output_dir / "tensorboard")


if __name__ == "__main__":
    main()
