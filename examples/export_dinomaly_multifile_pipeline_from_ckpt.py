"""Export Dinomaly multifile pipeline YAML + .pt from a PyTorch Lightning checkpoint.

Training writes checkpoints under ``<output_dir>/checkpoints/`` with ``save_last=True``:

- ``epoch=NN.ckpt`` — best epoch on ``metrics_anomaly/iou`` (only one kept when ``save_top_k=1``).
- ``last.ckpt`` — weights after the last *completed* training epoch (not necessarily best).

Use this to produce ``dinomaly_multifile_{cir|rgb}.{yaml,pt}`` in the same format as
``pipeline.save_to_file`` after training, without re-running fit.

Example (CIR, NPZ splits, last checkpoint)::

    cd /home/dev/anish/cuvis-ai-dinomaly
    uv run python examples/export_dinomaly_multifile_pipeline_from_ckpt.py \\
        --ckpt /mnt/data/cuvis_ai_outputs/dinomaly_cir_npz_50ep_w0/checkpoints/last.ckpt \\
        --output-dir /mnt/data/cuvis_ai_outputs/dinomaly_cir_npz_50ep_w0/trained_models_from_last \\
        --config configs/trainrun/dinomaly_multifile_cir.yaml \\
        --band cir
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import torch
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
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PipelineMetadata
from cuvis_ai_schemas.training import TrainingConfig
from loguru import logger
from omegaconf import OmegaConf

from cuvis_ai_dinomaly.data import MultiFileNpzDataModule


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    out: dict = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _infer_backend(splits_csv: Path) -> Literal["npz", "cu3s"]:
    if not splits_csv.is_file():
        return "cu3s"
    try:
        header = splits_csv.open(encoding="utf-8").readline()
    except OSError:
        return "cu3s"
    return "npz" if "npz_path" in header else "cu3s"


def build_pipeline_and_datamodule(
    *,
    cfg: object,
    run_root: Path,
    band_mode: Literal["rgb", "cir"],
) -> tuple[CuvisPipeline, object, MinMaxNormalizer, GradientTrainer]:
    """Match ``examples/dinomaly_multifile_train_common.py`` graph construction."""
    plugins_manifest = Path(__file__).resolve().parent / "plugins.yaml"
    registry = NodeRegistry()
    registry.load_plugins(str(plugins_manifest))
    DinomalyDetector = NodeRegistry.get("cuvis_ai_dinomaly.node.dinomaly_detector.DinomalyDetector")
    DinomalyTrainLossBridge = NodeRegistry.get(
        "cuvis_ai_dinomaly.node.dinomaly_train_loss_bridge.DinomalyTrainLossBridge",
    )

    splits_csv_path = Path(cfg.data.splits_csv)
    backend = _infer_backend(splits_csv_path)
    # Export runs a fresh StatisticalTrainer pass; NPZ dataset + spawn workers often
    # hit PicklingError (same as full training). Use a single-threaded loader here.
    common_loader_kwargs = {
        "splits_csv": str(splits_csv_path),
        "batch_size": int(cfg.data.batch_size),
        "num_workers": 0,
        "pin_memory": bool(cfg.data.get("pin_memory", True)),
        "persistent_workers": False,
        "worker_multiprocessing_context": str(
            cfg.data.get("worker_multiprocessing_context", "spawn")
        ),
    }
    if backend == "npz":
        datamodule = MultiFileNpzDataModule(**common_loader_kwargs)
    else:
        datamodule = MultiFileCu3sDataModule(
            **common_loader_kwargs,
            processing_mode=str(cfg.data.processing_mode),
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
        output_dir=str(run_root / "tensorboard"),
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

    device_s = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_s)
    pipeline.to(device)

    if normalizer.requires_initial_fit:
        logger.info("Statistical initialization (MinMaxNormalizer)...")
        StatisticalTrainer(pipeline=pipeline, datamodule=datamodule).fit()

    unfreeze_names = list(cfg.unfreeze_nodes) if "unfreeze_nodes" in cfg else ["dinomaly_detector"]
    pipeline.unfreeze_nodes_by_name(unfreeze_names)
    pipeline.to(device)

    training_cfg = TrainingConfig.from_dict(OmegaConf.to_container(cfg.training, resolve=True))
    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=[loss_bridge],
        metric_nodes=[metrics_node],
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
        monitors=[tb],
        callbacks=[],
    )

    return pipeline, datamodule, normalizer, grad_trainer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Lightning checkpoint, e.g. .../checkpoints/last.ckpt",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for dinomaly_multifile_*.yaml and .pt",
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Trainrun YAML used for the run (for graph + training section), e.g. configs/trainrun/dinomaly_multifile_cir.yaml",
    )
    p.add_argument(
        "--band",
        choices=("cir", "rgb"),
        default="cir",
        help="Must match training (pipeline name dinomaly_multifile_{band})",
    )
    p.add_argument("--device", type=str, default=None, help="cuda | cpu")
    p.add_argument(
        "--strict-load",
        action="store_true",
        help="Use strict=True when loading pipeline_modules into GradientTrainer",
    )
    p.add_argument(
        "--print-keys",
        action="store_true",
        help="Print checkpoint / state_dict key info and exit",
    )
    args = p.parse_args()

    ckpt_path = args.ckpt.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_root = ckpt_path.parent.parent.resolve()

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)
    logger.info("Device: {}", device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if args.print_keys:
        print("checkpoint keys:", sorted(ckpt.keys()))
        sd = ckpt.get("state_dict", {})
        prefixes = sorted({k.split(".", 2)[0] for k in sd if k})
        print("state_dict top-level prefixes:", prefixes)
        pm = [k for k in sd if k.startswith("pipeline_modules.")]
        print(f"pipeline_modules keys: {len(pm)} (showing up to 25)")
        for k in pm[:25]:
            print(" ", k)
        return

    cfg = OmegaConf.load(cfg_path)
    pipeline, datamodule, _normalizer, grad_trainer = build_pipeline_and_datamodule(
        cfg=cfg,
        run_root=run_root,
        band_mode=args.band,
    )

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    batch = _batch_to_device(batch, device)
    # VAL avoids Dinomaly TRAIN loss path (hooks on tensors that need grad); still builds the full eval graph.
    ctx = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0, global_step=0)
    logger.info("Warmup forward (materialize Dinomaly submodules)...")
    with torch.no_grad():
        pipeline.forward(batch=batch, context=ctx)

    state = ckpt["state_dict"]
    filtered = {k: v for k, v in state.items() if k.startswith("pipeline_modules.")}
    missing, unexpected = grad_trainer.load_state_dict(filtered, strict=args.strict_load)
    if missing:
        logger.warning("load_state_dict missing ({}): {}", len(missing), list(missing)[:20])
    if unexpected:
        logger.warning(
            "load_state_dict unexpected ({}): {}", len(unexpected), list(unexpected)[:20]
        )

    yaml_path = out_dir / f"{pipeline.name}.yaml"
    pipeline.save_to_file(
        str(yaml_path),
        metadata=PipelineMetadata(
            name=pipeline.name,
            description=(
                f"Exported from Lightning checkpoint {ckpt_path.name} "
                f"({args.band.upper()} Dinomaly multifile; weights match checkpoint)."
            ),
            tags=["dinomaly", "anomalib", "multifile", args.band, "exported_from_ckpt"],
            author="cuvis.ai",
        ),
    )
    logger.info("Wrote {} and matching .pt", yaml_path)


if __name__ == "__main__":
    main()
