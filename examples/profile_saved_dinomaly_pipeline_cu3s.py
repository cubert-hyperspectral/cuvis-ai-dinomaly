"""Profile saved Dinomaly pipeline inference time per node on CU3S test frames.

Runs a saved Dinomaly pipeline (YAML + .pt) on a small number of CU3S test images
and reports:
- per-frame end-to-end latency (ms)
- per-node aggregated timings from CuvisPipeline profiling

Example:
    cd /home/dev/anish/cuvis-ai-dinomaly
    uv run python examples/profile_saved_dinomaly_pipeline_cu3s.py \
      --pipeline-yaml /mnt/data/cuvis_ai_outputs/dinomaly_cir_npz_50ep_w0/trained_models_best/dinomaly_multifile_cir.yaml \
      --pipeline-pt /mnt/data/cuvis_ai_outputs/dinomaly_cir_npz_50ep_w0/trained_models_best/dinomaly_multifile_cir.pt \
      --splits-csv /home/dev/anish/cuvis-ai/lentils_splits.csv \
      --num-images 10
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from cuvis_ai.data import MultiFileCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from loguru import logger

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))


def _move_batch(batch: dict, device: torch.device) -> dict:
    out: dict = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline-yaml", type=Path, required=True)
    ap.add_argument("--pipeline-pt", type=Path, required=True)
    ap.add_argument(
        "--plugins",
        type=Path,
        default=None,
        help="Plugin manifest (default: examples/plugins.yaml)",
    )
    ap.add_argument("--splits-csv", type=Path, default=Path("/home/dev/anish/cuvis-ai/lentils_splits.csv"))
    ap.add_argument("--num-images", type=int, default=10)
    ap.add_argument("--warmup-images", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--processing-mode", type=str, default="Reflectance")
    ap.add_argument("--device", type=str, default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--strict-weights", action="store_true")
    args = ap.parse_args()

    yaml_path = args.pipeline_yaml.resolve()
    pt_path = args.pipeline_pt.resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    if not pt_path.is_file():
        raise FileNotFoundError(pt_path)

    plugins_path = args.plugins.resolve() if args.plugins else (_EXAMPLES / "plugins.yaml")
    if not plugins_path.is_file():
        raise FileNotFoundError(plugins_path)

    if not args.splits_csv.is_file():
        raise FileNotFoundError(args.splits_csv)

    total_needed = args.warmup_images + args.num_images
    if total_needed <= 0:
        raise ValueError("warmup-images + num-images must be > 0")

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)
    logger.info("Device: {}", device)
    logger.info("Pipeline: {} + {}", yaml_path, pt_path)

    registry = NodeRegistry()
    registry.load_plugins(str(plugins_path))

    pipeline = CuvisPipeline.load_pipeline(
        yaml_path,
        weights_path=str(pt_path),
        device=device_s,
        strict_weight_loading=args.strict_weights,
        node_registry=registry,
    )
    pipeline.torch_layers.eval()

    datamodule = MultiFileCu3sDataModule(
        splits_csv=str(args.splits_csv),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        processing_mode=args.processing_mode,
    )
    datamodule.setup(stage="test")
    if datamodule.test_ds is None or len(datamodule.test_ds) == 0:
        raise RuntimeError("No test split found in splits CSV.")
    if len(datamodule.test_ds) < total_needed:
        raise RuntimeError(
            f"Not enough test frames: need {total_needed}, have {len(datamodule.test_ds)}."
        )

    loader = datamodule.test_dataloader()
    records = datamodule.test_ds.records

    pipeline.set_profiling(
        enabled=True,
        reset=True,
        skip_first_n=0,
        synchronize_cuda=(device.type == "cuda"),
    )

    frame_ms: list[float] = []
    processed = 0
    timed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if processed >= total_needed:
                break
            batch = _move_batch(batch, device)
            bsz = int(batch["cube"].shape[0])
            for i in range(bsz):
                if processed >= total_needed:
                    break
                ds_idx = processed
                sub = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        sub[k] = v[i : i + 1]
                    elif isinstance(v, list):
                        sub[k] = v[i : i + 1]
                    else:
                        sub[k] = v

                ctx = Context(
                    stage=ExecutionStage.INFERENCE,
                    epoch=0,
                    batch_idx=batch_idx,
                    global_step=ds_idx,
                )

                t0 = time.perf_counter()
                _ = pipeline.forward(batch=sub, context=ctx)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt_ms = (time.perf_counter() - t0) * 1000.0

                if processed >= args.warmup_images:
                    frame_ms.append(dt_ms)
                    timed += 1
                    rec = records[ds_idx]
                    cu3s_path = rec.get("cu3s_path", "unknown")
                    logger.info("[{:02d}/{:02d}] {} -> {} ms", timed, args.num_images, cu3s_path, _fmt(dt_ms))

                processed += 1

                if processed == args.warmup_images:
                    # Clear warmup traces so reported node timings only include measured frames.
                    pipeline.reset_profiling()

    stats = pipeline.get_profiling_summary(stage=ExecutionStage.INFERENCE)
    stats_sorted = sorted(stats, key=lambda s: float(s.total_ms), reverse=True)

    print("\n=== Frame Latency (ms) ===")
    print(f"frames: {len(frame_ms)}")
    print(f"mean:   {_fmt(statistics.mean(frame_ms))}")
    print(f"median: {_fmt(statistics.median(frame_ms))}")
    print(f"min:    {_fmt(min(frame_ms))}")
    print(f"max:    {_fmt(max(frame_ms))}")

    print("\n=== Per-Node Profiling (INFERENCE) ===")
    print(f"{'node':40s} {'count':>7s} {'mean_ms':>10s} {'total_ms':>10s} {'max_ms':>10s}")
    for s in stats_sorted:
        print(
            f"{str(s.node_name):40.40s} "
            f"{int(s.count):7d} "
            f"{_fmt(float(s.mean_ms)):>10s} "
            f"{_fmt(float(s.total_ms)):>10s} "
            f"{_fmt(float(s.max_ms)):>10s}"
        )


if __name__ == "__main__":
    main()

