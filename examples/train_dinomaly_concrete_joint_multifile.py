"""Train Dinomaly + Concrete selector jointly with distinctness regularization.

Standalone script (does not use the shared RGB/CIR training common module).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from cuvis_ai.data import MultiFileCu3sDataModule
from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node.channel_mixer import ConcreteChannelMixer
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PipelineMetadata, PortSpec
from cuvis_ai_schemas.training import (
    CallbacksConfig,
    ModelCheckpointConfig,
    TrainingConfig,
)
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from cuvis_ai_dinomaly.data import MultiFileNpzDataModule


class TrainOnlyDistinctnessLoss(Node):
    """Pairwise cosine repulsion loss for selector weights, train stage only."""

    INPUT_SPECS = {
        "selection_weights": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="Selector weights [C_out, C_in].",
        )
    }
    OUTPUT_SPECS = {"loss": PortSpec(dtype=torch.float32, shape=(), description="Distinctness loss")}

    def __init__(self, weight: float = 0.1, eps: float = 1e-6, **kwargs: Any) -> None:
        self.weight = float(weight)
        self.eps = float(eps)
        super().__init__(execution_stages={ExecutionStage.TRAIN}, weight=self.weight, eps=self.eps, **kwargs)

    def forward(self, selection_weights: Tensor, **_: Any) -> dict[str, Tensor]:
        w_norm = F.normalize(selection_weights, p=2, dim=-1, eps=self.eps)
        sim = w_norm @ w_norm.T
        upper = torch.triu(sim, diagonal=1)
        vals = upper[upper != 0]
        if vals.numel() == 0:
            loss = torch.zeros((), device=w_norm.device, dtype=w_norm.dtype)
        else:
            loss = vals.mean()
        return {"loss": self.weight * loss}


@hydra.main(config_path="../configs", config_name="trainrun/dinomaly_multifile_concrete_joint", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("=== Dinomaly + Concrete selector (joint + distinctness, standalone) ===")

    plugins_manifest = Path(__file__).resolve().parent / "plugins.yaml"
    registry = NodeRegistry()
    registry.load_plugins(str(plugins_manifest))
    DinomalyDetector = NodeRegistry.get("cuvis_ai_dinomaly.node.dinomaly_detector.DinomalyDetector")
    DinomalyTrainLossBridge = NodeRegistry.get(
        "cuvis_ai_dinomaly.node.dinomaly_train_loss_bridge.DinomalyTrainLossBridge",
    )

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = "cu3s"
    splits_csv_path = Path(cfg.data.splits_csv)
    if splits_csv_path.is_file():
        header = splits_csv_path.open(encoding="utf-8").readline()
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
        datamodule = MultiFileCu3sDataModule(**common_loader_kwargs, processing_mode=cfg.data.processing_mode)
    datamodule.setup(stage="fit")

    first_batch = next(iter(datamodule.train_dataloader()))
    input_channels = int(first_batch["cube"].shape[-1])

    dcfg = cfg.dinomaly
    pipeline = CuvisPipeline("dinomaly_multifile_concrete_joint")
    data_node = LentilsAnomalyDataNode(normal_class_ids=[0])
    normalizer = MinMaxNormalizer(eps=1e-6, use_running_stats=True, max_initialization_frames=cfg.get("minmax_init_frames", None))
    selector = ConcreteChannelMixer(
        input_channels=input_channels,
        output_channels=int(cfg.get("concrete", {}).get("output_channels", 3)),
        tau_start=float(cfg.get("concrete", {}).get("tau_start", 10.0)),
        tau_end=float(cfg.get("concrete", {}).get("tau_end", 0.1)),
        max_epochs=int(cfg.training.trainer.max_epochs),
        use_hard_inference=bool(cfg.get("concrete", {}).get("use_hard_inference", True)),
        eps=float(cfg.get("concrete", {}).get("eps", 1e-6)),
        name="concrete_selector",
    )
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
    distinctness = TrainOnlyDistinctnessLoss(
        weight=float(cfg.get("concrete", {}).get("distinctness_weight", 0.1)),
        eps=float(cfg.get("concrete", {}).get("distinctness_eps", 1e-6)),
        name="distinctness_loss",
    )
    decider = QuantileBinaryDecider(quantile=0.995, name="decider")
    metrics_node = AnomalyDetectionMetrics(name="metrics_anomaly")
    tb = TensorBoardMonitorNode(output_dir=str(output_dir / "tensorboard"), run_name=pipeline.name)

    pipeline.connect(
        (data_node.outputs.cube, normalizer.data),
        (normalizer.normalized, selector.data),
        (selector.rgb, dinomaly.rgb_image),
        (dinomaly.outputs.training_loss, loss_bridge.raw_loss),
        (selector.selection_weights, distinctness.selection_weights),
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
        StatisticalTrainer(pipeline=pipeline, datamodule=datamodule).fit()

    unfreeze = list(cfg.unfreeze_nodes) if "unfreeze_nodes" in cfg else ["dinomaly_detector", "concrete_selector"]
    if "concrete_selector" not in unfreeze:
        unfreeze.append("concrete_selector")
    if "dinomaly_detector" not in unfreeze:
        unfreeze.append("dinomaly_detector")
    pipeline.unfreeze_nodes_by_name(unfreeze)
    pipeline.to(device)
    logger.info("Unfrozen: {}", unfreeze)

    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=[loss_bridge, distinctness],
        metric_nodes=[metrics_node],
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
        monitors=[tb],
    )
    grad_trainer.fit()
    grad_trainer.datamodule.setup(stage="test")
    grad_trainer.validate(ckpt_path="best")
    grad_trainer.test(ckpt_path="best")

    results_dir = output_dir / "trained_models"
    results_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = results_dir / f"{pipeline.name}.yaml"
    pipeline.save_to_file(
        str(pipeline_path),
        metadata=PipelineMetadata(
            name=pipeline.name,
            description="Dinomaly + Concrete selector joint training with train-only distinctness regularizer.",
            tags=["dinomaly", "concrete", "joint", "distinctness"],
            author="cuvis.ai",
        ),
    )
    logger.info("Pipeline saved: {}", pipeline_path)


if __name__ == "__main__":
    main()

