"""Run a saved Dinomaly multifile pipeline on the **test** split and write NPZ predictions.

Mirrors ``cuvis-ai/examples/adaclip/run_saved_pipeline_test_npz.py`` for the Dinomaly graph:
load plugins, ``CuvisPipeline.load_pipeline(yaml, weights_path=pt, ...)``, then iterate
``test_dataloader()`` with ``ExecutionStage.TEST``.

**NPZ keys (typical)**

- ``anomaly_map`` — ``dinomaly_detector.scores`` without batch dim ``[H, W]`` or ``[H, W, 1]`` squeezed.
- ``anomaly_score`` — image-level score ``[1]`` or scalar.
- ``binary_decisions`` — decider output (float32 for portability).
- ``gt_mask`` — int32 category mask (same as dataloader).
- ``rgb_image`` — CIR/RGB selector output before Dinomaly internal resize.
- ``metric_names``, ``metric_values`` — per-batch ``AnomalyDetectionMetrics`` (TEST).
- ``mesu_index`` — ``image_id`` from splits CSV.
- ``source_npz_path`` or ``cu3s_path`` — frame path (Unicode 0-d array), depending on backend.

Writes ``manifest.jsonl`` and ``test_metrics_summary.json`` (means of node metrics across
frames + simple per-image IoU / precision / recall / F1 from saved decisions vs GT).

Example::

    cd /home/dev/anish/cuvis-ai-dinomaly
    uv run python examples/run_saved_dinomaly_pipeline_test_npz.py \\
        --pipeline-yaml /mnt/data/cuvis_ai_outputs/dinomaly_cir_npz_50ep_w0/trained_models/dinomaly_multifile_cir.yaml \\
        --pipeline-pt /mnt/data/cuvis_ai_outputs/dinomaly_cir_npz_50ep_w0/trained_models/dinomaly_multifile_cir.pt \\
        --splits-csv /home/dev/anish/cuvis-ai-dinomaly/diagnostics/lentils_splits_npz_full.csv \\
        --output-dir /mnt/data/cuvis_ai_outputs/dinomaly_cir_npz_50ep_w0/test_predictions_npz
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from cuvis_ai.data import MultiFileCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.utils.graph_helper import restructure_output_to_node_dict
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from loguru import logger

from cuvis_ai_dinomaly.data import MultiFileNpzDataModule


def _to_numpy(x: torch.Tensor | None) -> np.ndarray | None:
    if x is None:
        return None
    if not torch.is_tensor(x):
        return None
    return x.detach().float().cpu().numpy()


def _sample_np(
    x: torch.Tensor | None,
    sample_idx: int,
    *,
    expected_batch_size: int | None = None,
) -> np.ndarray | None:
    if x is None or not torch.is_tensor(x):
        return None
    if x.ndim == 0:
        return _to_numpy(x) if sample_idx == 0 else None
    if expected_batch_size is not None:
        if x.shape[0] == expected_batch_size:
            if sample_idx < expected_batch_size:
                return _to_numpy(x[sample_idx])
            return None
        return _to_numpy(x) if sample_idx == 0 else None
    if x.shape[0] <= sample_idx:
        return None
    return _to_numpy(x[sample_idx])


def _metrics_to_arrays(metrics: object) -> tuple[np.ndarray, np.ndarray]:
    if not metrics:
        return np.array([], dtype=str), np.array([], dtype=np.float32)
    names: list[str] = []
    values: list[float] = []
    for m in metrics:
        name = getattr(m, "name", None)
        val = getattr(m, "value", None)
        if name is None:
            continue
        if torch.is_tensor(val):
            val = val.detach().float().cpu().item()
        elif val is not None:
            val = float(val)
        else:
            continue
        names.append(str(name))
        values.append(val)
    return np.array(names, dtype=str), np.array(values, dtype=np.float32)


def _move_batch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _node_dict_ports(node_out: dict[str, object], node_name: str) -> dict:
    v = node_out.get(node_name)
    return v if isinstance(v, dict) else {}


def _pick_dinomaly_outputs(node_out: dict[str, object]) -> dict:
    for name in ("dinomaly_detector", "DinomalyDetector"):
        d = _node_dict_ports(node_out, name)
        if d.get("scores") is not None:
            return d
    for _k, v in node_out.items():
        if isinstance(v, dict) and v.get("scores") is not None:
            return v
    return {}


def _pick_decider_outputs(node_out: dict[str, object]) -> dict:
    for name in ("decider", "quantile_decider", "QuantileBinaryDecider"):
        d = _node_dict_ports(node_out, name)
        if d.get("decisions") is not None:
            return d
    for _k, v in node_out.items():
        if isinstance(v, dict) and v.get("decisions") is not None:
            return v
    return {}


def _pick_metrics_outputs(node_out: dict[str, object]) -> dict:
    for name in ("metrics_anomaly", "anomaly_metrics", "detection_metrics"):
        d = _node_dict_ports(node_out, name)
        if d.get("metrics") is not None:
            return d
    for _k, v in node_out.items():
        if isinstance(v, dict) and v.get("metrics") is not None:
            return v
    return {}


def _pick_selector_outputs(node_out: dict[str, object]) -> dict:
    for name in ("cir_selector", "rgb_selector", "CIRSelector", "FixedWavelengthSelector"):
        d = _node_dict_ports(node_out, name)
        if d.get("rgb_image") is not None:
            return d
    for _k, v in node_out.items():
        if isinstance(v, dict) and v.get("rgb_image") is not None:
            return v
    return {}


def _gt_anomaly_bool(mask_hw: np.ndarray) -> np.ndarray:
    """Binary anomaly map: category 0 = normal (matches typical lentils normal_class_ids=[0])."""
    return (mask_hw.astype(np.int32) != 0)


def _binary_prf_iou(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    p = pred.astype(bool).ravel()
    g = gt.astype(bool).ravel()
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    inter = tp
    union = int(np.logical_or(p, g).sum())
    iou = inter / union if union > 0 else (1.0 if inter == 0 else 0.0)
    return {"precision": prec, "recall": rec, "f1_score": f1, "iou": iou}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline-yaml", type=Path, required=True)
    p.add_argument("--pipeline-pt", type=Path, required=True)
    p.add_argument(
        "--plugins",
        type=Path,
        default=None,
        help="Plugin manifest (default: examples/plugins.yaml next to this file)",
    )
    p.add_argument("--splits-csv", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--processing-mode", type=str, default="Reflectance")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--strict-weights", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="If >0, only process this many test frames.")
    args = p.parse_args()

    yaml_path = args.pipeline_yaml.resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    pt_path = args.pipeline_pt.resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(pt_path)

    examples_dir = Path(__file__).resolve().parent
    plugins_path = args.plugins.resolve() if args.plugins else (examples_dir / "plugins.yaml")
    if not plugins_path.is_file():
        raise FileNotFoundError(plugins_path)

    splits_path = Path(args.splits_csv)
    if not splits_path.is_file():
        raise FileNotFoundError(splits_path)

    header = splits_path.open(encoding="utf-8").readline()
    backend = "npz" if "npz_path" in header else "cu3s"

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)
    logger.info("Device: {}, backend: {}", device, backend)

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

    common = dict(
        splits_csv=str(splits_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    if backend == "npz":
        datamodule = MultiFileNpzDataModule(**common)
    else:
        datamodule = MultiFileCu3sDataModule(
            **common,
            processing_mode=args.processing_mode,
        )

    datamodule.setup(stage="test")
    if datamodule.test_ds is None or len(datamodule.test_ds) == 0:
        raise RuntimeError("No test split in splits CSV (or test_ds empty).")

    loader = datamodule.test_dataloader()
    records = datamodule.test_ds.records
    manifest_path = out_dir / "manifest.jsonl"

    n_test = len(datamodule.test_ds)
    if args.limit > 0:
        n_test = min(n_test, args.limit)

    node_metric_accum: dict[str, list[float]] = defaultdict(list)
    per_image_stats: list[dict[str, float]] = []

    global_offset = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for batch_idx, batch in enumerate(loader):
            if args.limit > 0 and global_offset >= args.limit:
                break
            batch = _move_batch(batch, device)
            bsz = int(batch["cube"].shape[0])
            if args.limit > 0:
                bsz = min(bsz, args.limit - global_offset)

            ctx = Context(
                stage=ExecutionStage.TEST,
                epoch=0,
                batch_idx=batch_idx,
                global_step=global_offset,
            )
            with torch.no_grad():
                raw_out = pipeline.forward(batch=batch, context=ctx)
            node_out = restructure_output_to_node_dict(raw_out)

            dinomaly = _pick_dinomaly_outputs(node_out)
            decider = _pick_decider_outputs(node_out)
            metrics_node = _pick_metrics_outputs(node_out)
            selector = _pick_selector_outputs(node_out)

            for i in range(bsz):
                ds_idx = global_offset + i
                rec = records[ds_idx]

                if backend == "npz":
                    frame_path = str(rec["npz_path"])
                    path_key = "source_npz_path"
                else:
                    frame_path = str(rec["cu3s_path"])
                    path_key = "cu3s_path"
                stem = Path(frame_path).stem
                npz_name = f"{ds_idx:05d}_{stem}.npz"
                npz_path = out_dir / npz_name

                scores = _sample_np(dinomaly.get("scores"), i, expected_batch_size=bsz)
                anomaly_map = scores
                if anomaly_map is not None and anomaly_map.ndim == 3 and anomaly_map.shape[-1] == 1:
                    anomaly_map = anomaly_map[..., 0]

                ascore = _sample_np(dinomaly.get("anomaly_score"), i, expected_batch_size=bsz)

                decisions = _sample_np(decider.get("decisions"), i, expected_batch_size=bsz)
                if decisions is not None and decisions.ndim == 3 and decisions.shape[-1] == 1:
                    decisions = decisions[..., 0]

                mask = _sample_np(batch.get("mask"), i, expected_batch_size=bsz)
                rgb = _sample_np(selector.get("rgb_image"), i, expected_batch_size=bsz)

                m_names, m_vals = _metrics_to_arrays(metrics_node.get("metrics"))
                for ni, nv in zip(m_names.tolist(), m_vals.tolist(), strict=False):
                    node_metric_accum[str(ni)].append(float(nv))

                pi: dict[str, float] = {}
                if decisions is not None and mask is not None:
                    gt = _gt_anomaly_bool(mask)
                    pred = decisions.astype(bool) if decisions.dtype != bool else decisions
                    pi = _binary_prf_iou(pred, gt)
                    per_image_stats.append(pi)

                save_kw: dict[str, np.ndarray] = {
                    "anomaly_map": anomaly_map.astype(np.float32)
                    if anomaly_map is not None
                    else np.array([], np.float32),
                    "binary_decisions": decisions.astype(np.float32)
                    if decisions is not None
                    else np.array([], np.float32),
                    "gt_mask": mask.astype(np.int32) if mask is not None else np.array([], np.int32),
                    "mesu_index": np.array(int(batch["mesu_index"][i].item()), dtype=np.int64),
                    path_key: np.array(frame_path, dtype=str),
                }
                if ascore is not None:
                    save_kw["anomaly_score"] = np.asarray(ascore, dtype=np.float32).reshape(-1)
                if rgb is not None:
                    save_kw["rgb_image"] = rgb.astype(np.float32)
                if m_names.size:
                    save_kw["metric_names"] = m_names
                    save_kw["metric_values"] = m_vals

                np.savez_compressed(npz_path, **save_kw)
                man_obj: dict[str, object] = {
                    "npz": str(npz_path),
                    "frame_path": frame_path,
                    "mesu_index": int(batch["mesu_index"][i].item()),
                }
                if backend == "npz" and rec.get("cu3s_path"):
                    man_obj["cu3s_path"] = str(rec["cu3s_path"])
                mf.write(json.dumps(man_obj) + "\n")

            global_offset += bsz
            if global_offset % 50 == 0 or global_offset == n_test:
                logger.info("Progress {}/{} test frames", global_offset, n_test)

            if args.limit > 0 and global_offset >= args.limit:
                break

    summary: dict[str, object] = {
        "n_frames": int(global_offset),
        "backend": backend,
        "splits_csv": str(splits_path),
        "pipeline_yaml": str(yaml_path),
        "pipeline_pt": str(pt_path),
        "node_metrics_mean_over_frames": {k: float(np.mean(v)) for k, v in sorted(node_metric_accum.items())},
    }
    if per_image_stats:
        keys = ["precision", "recall", "f1_score", "iou"]
        summary["per_image_mean"] = {
            k: float(np.mean([s[k] for s in per_image_stats])) for k in keys
        }

    summary_path = out_dir / "test_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote {}", summary_path)
    logger.info("Done. NPZ in {} (manifest: {})", out_dir, manifest_path)


if __name__ == "__main__":
    main()
