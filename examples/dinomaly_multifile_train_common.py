"""Shared MultiFileCu3s + Dinomaly gradient training (RGB or CIR selector)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from cuvis_ai.data import MultiFileCu3sDataModule
from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node.channel_selector import CIRSelector, FixedWavelengthSelector
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.pipeline import PipelineMetadata
from cuvis_ai_schemas.training import (
    CallbacksConfig,
    ModelCheckpointConfig,
    TrainingConfig,
)
from cuvis_ai_core.training.config import create_callbacks_from_config
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cuvis_ai_dinomaly.data import MultiFileNpzDataModule


class _LastCheckpointAuditCallback(Callback):
    """Log and copy Lightning's `last_model_path` at end of training (on_train_end)."""

    def __init__(self, output_dir: Path) -> None:
        super().__init__()
        self._output_dir = output_dir

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        ck_dir = self._output_dir / "checkpoints"
        ck_dir.mkdir(parents=True, exist_ok=True)
        audit: dict[str, object] = {"callbacks": []}
        for c in trainer.callbacks:
            if not isinstance(c, ModelCheckpoint):
                continue
            last_p = getattr(c, "last_model_path", None)
            entry: dict[str, object] = {
                "dirpath": str(c.dirpath) if c.dirpath else None,
                "last_model_path": str(last_p) if last_p else None,
                "best_model_path": str(c.best_model_path) if getattr(c, "best_model_path", None) else None,
            }
            if last_p and Path(last_p).is_file():
                ck = torch.load(last_p, map_location="cpu", weights_only=False)
                entry["epoch_in_checkpoint"] = ck.get("epoch")
                entry["global_step_in_checkpoint"] = ck.get("global_step")
                dest = ck_dir / f"last_on_disk_epoch{ck.get('epoch', 'unknown')}.ckpt"
                shutil.copy2(last_p, dest)
                entry["copied_to"] = str(dest)
            audit["callbacks"].append(entry)
        audit_path = ck_dir / "last_ckpt_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        logger.info("Last-checkpoint audit written to {}", audit_path)
        for entry in audit["callbacks"]:
            logger.info(
                "Checkpoint audit: last_model_path={} epoch_in_file={}",
                entry.get("last_model_path"),
                entry.get("epoch_in_checkpoint"),
            )


def run_dinomaly_multifile_training(
    cfg: DictConfig,
    *,
    band_mode: Literal["rgb", "cir"],
    run_title: str,
    eval_mode: Literal["best", "last_epoch"] | None = None,
) -> None:
    """Build pipeline, wire DinomalyTrainLossBridge explicitly, train with GradientTrainer.

    Parameters
    ----------
    eval_mode
        ``best`` (default): ``validate``/``test`` reload the best checkpoint; saved ``.pt`` matches best.
        ``last_epoch``: after ``fit``, ``validate``/``test`` use ``ckpt_path=None`` (in-memory last epoch).
        Saved pipeline goes to ``trained_models_last_epoch/``. Also set ``eval_mode`` in Hydra as ``+eval_mode=last_epoch``.
    """
    logger.info(run_title)

    plugins_manifest = Path(__file__).resolve().parent / "plugins.yaml"
    registry = NodeRegistry()
    registry.load_plugins(str(plugins_manifest))
    DinomalyDetector = NodeRegistry.get("cuvis_ai_dinomaly.node.dinomaly_detector.DinomalyDetector")
    DinomalyTrainLossBridge = NodeRegistry.get(
        "cuvis_ai_dinomaly.node.dinomaly_train_loss_bridge.DinomalyTrainLossBridge",
    )
    logger.info("Dinomaly plugin registered")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Infer datalake backend:
    # - CU3S: splits CSV without an `npz_path` column
    # - NPZ:  splits CSV includes an `npz_path` column
    #
    # We intentionally avoid requiring a `data.backend` config key to keep Hydra schemas unchanged.
    backend_cfg = cfg.data.get("backend", None)
    if backend_cfg is None:
        splits_csv_path = Path(cfg.data.splits_csv)
        backend = "cu3s"
        if splits_csv_path.is_file():
            try:
                header = splits_csv_path.open(encoding="utf-8").readline()
                if "npz_path" in header:
                    backend = "npz"
            except Exception:
                backend = "cu3s"
    else:
        backend = str(backend_cfg).lower()
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
    pipeline = CuvisPipeline(f"dinomaly_multifile_{band_mode}")

    data_node = LentilsAnomalyDataNode(normal_class_ids=[0])
    normalizer = MinMaxNormalizer(
        eps=1e-6,
        use_running_stats=True,
        max_initialization_frames=cfg.get("minmax_init_frames", None),
    )

    if band_mode == "rgb":
        selector = FixedWavelengthSelector(
            target_wavelengths=(650.0, 550.0, 450.0),
            name="rgb_selector",
        )
    else:
        selector = CIRSelector(
            nir_nm=860.0,
            red_nm=670.0,
            green_nm=560.0,
            norm_mode="running",
            running_warmup_frames=0,
            freeze_running_bounds_after_frames=20,
            name="cir_selector",
        )
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
    logger.info("Moving pipeline to device: %s", device)
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

    mode = str(eval_mode if eval_mode is not None else cfg.get("eval_mode", "best")).lower()
    if mode not in ("best", "last_epoch"):
        raise ValueError(f"eval_mode must be 'best' or 'last_epoch', got {mode!r}")
    extra_callbacks: list[Callback] = []
    if mode == "last_epoch":
        extra_callbacks.append(_LastCheckpointAuditCallback(output_dir))

    if normalizer.requires_initial_fit:
        logger.info("Phase 1: Statistical initialization (MinMaxNormalizer)...")
        StatisticalTrainer(pipeline=pipeline, datamodule=datamodule).fit()

    unfreeze_names = list(cfg.unfreeze_nodes) if "unfreeze_nodes" in cfg else ["dinomaly_detector"]
    pipeline.unfreeze_nodes_by_name(unfreeze_names)
    logger.info("Unfrozen: %s", unfreeze_names)
    pipeline.to(device)

    n_train = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in pipeline.parameters())
    logger.info(
        "Trainable: %s / %s params (%.2f%%)",
        f"{n_train:,}",
        f"{n_total:,}",
        100.0 * n_train / max(n_total, 1),
    )

    merged_callbacks: list | None = None
    if extra_callbacks:
        merged_callbacks = list(create_callbacks_from_config(training_cfg.trainer.callbacks)) + extra_callbacks

    logger.info("Gradient training (Dinomaly bottleneck + decoder)...")
    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=[loss_bridge],
        metric_nodes=[metrics_node],
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
        monitors=[tb],
        callbacks=merged_callbacks,
    )
    grad_trainer.fit()

    # Ensure test split is loaded (initial setup was only fit for train/val).
    grad_trainer.datamodule.setup(stage="test")

    if mode == "best":
        logger.info("Validation with best checkpoint...")
        grad_trainer.validate(ckpt_path="best")
        logger.info("Test with best checkpoint...")
        grad_trainer.test(ckpt_path="best")
        results_dir = output_dir / "trained_models"
        save_desc = (
            f"Dinomaly (Anomalib) on multi-file lentils, {band_mode.upper()} selector, "
            "with explicit DinomalyTrainLossBridge in the graph."
        )
        save_tags = ["dinomaly", "anomalib", "multifile", band_mode]
    else:
        # In-memory weights after fit() are the last completed epoch; do not reload best.
        logger.info("Validation with last epoch (in-memory weights, ckpt_path=None)...")
        grad_trainer.validate(ckpt_path=None)
        logger.info("Test with last epoch (in-memory weights, ckpt_path=None)...")
        grad_trainer.test(ckpt_path=None)
        results_dir = output_dir / "trained_models_last_epoch"
        save_desc = (
            f"Dinomaly (Anomalib) on multi-file lentils, {band_mode.upper()} selector; "
            "weights from last training epoch (val/test used ckpt_path=None)."
        )
        save_tags = ["dinomaly", "anomalib", "multifile", band_mode, "last_epoch_eval"]

    results_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = results_dir / f"{pipeline.name}.yaml"
    pipeline.save_to_file(
        str(pipeline_path),
        metadata=PipelineMetadata(
            name=pipeline.name,
            description=save_desc,
            tags=save_tags,
            author="cuvis.ai",
        ),
    )
    logger.info("Pipeline saved: %s", pipeline_path)
    logger.info("TensorBoard: uv run tensorboard --logdir=%s", output_dir / "tensorboard")
