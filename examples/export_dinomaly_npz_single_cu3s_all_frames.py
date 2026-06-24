"""Export Dinomaly saved pipeline to NPZ for every measurement in one .cu3s.

Uses ``Cu3sDataModule.setup(stage='predict')`` (all frames).
NPZ keys mirror ``run_saved_dinomaly_pipeline_test_npz.py`` plus ``image_score_topk``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.utils.graph_helper import restructure_output_to_node_dict
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_dataloader.data import Cu3sDataModule
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from loguru import logger


def two_stage_gate_image_score(score_map: np.ndarray, top_k_fraction: float) -> float:
    """Mean of top-k fraction of flattened scores (ceil k), same as cuvis-ai lentils_eval."""
    vals = score_map.astype(np.float32, copy=False).ravel()
    n = vals.size
    k = max(1, int(math.ceil(n * top_k_fraction)))
    if k >= n:
        return float(vals.mean())
    return float(np.partition(vals, n - k)[n - k :].mean())


def _to_numpy(x: torch.Tensor | None) -> np.ndarray | None:
    if x is None or not torch.is_tensor(x):
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
        values.append(float(val))
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
    for name in (
        "cir_selector",
        "rgb_selector",
        "CIRSelector",
        "FixedWavelengthSelector",
        "concrete_selector",
        "ConcreteChannelMixer",
    ):
        d = _node_dict_ports(node_out, name)
        if d.get("rgb_image") is not None:
            return d
    for _k, v in node_out.items():
        if isinstance(v, dict) and v.get("rgb_image") is not None:
            return v
    return {}


def main() -> None:
    examples_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cu3s-file-path", type=Path, required=True)
    p.add_argument("--annotation-json-path", type=Path, default=None)
    p.add_argument("--pipeline-yaml", type=Path, required=True)
    p.add_argument("--pipeline-pt", type=Path, required=True)
    p.add_argument(
        "--plugins",
        type=Path,
        default=examples_dir / "plugins.yaml",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--processing-mode", type=str, default="Reflectance")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--strict-weights", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--top-k-fraction", type=float, default=0.001)
    args = p.parse_args()

    cu3s_path = args.cu3s_file_path.resolve()
    if not cu3s_path.is_file():
        raise FileNotFoundError(cu3s_path)
    ann = args.annotation_json_path
    if ann is None:
        ann = cu3s_path.with_suffix(".json")
    ann = ann.resolve() if ann.is_file() else None

    yaml_path = args.pipeline_yaml.resolve()
    pt_path = args.pipeline_pt.resolve()
    for path in (yaml_path, pt_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    plugins_path = args.plugins.resolve()
    if not plugins_path.is_file():
        raise FileNotFoundError(plugins_path)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

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

    dm = Cu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        annotation_json_path=str(ann) if ann else None,
        batch_size=args.batch_size,
        processing_mode=args.processing_mode,
    )
    dm.setup(stage="predict")
    loader = dm.predict_dataloader()

    cu3s_str = str(cu3s_path)
    stem = cu3s_path.stem
    manifest_path = out_dir / "manifest.jsonl"
    meta_path = out_dir / "export_meta.json"
    n_total = len(loader.dataset)
    n_predict = n_total if args.limit <= 0 else min(n_total, args.limit)

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
                stage=ExecutionStage.INFERENCE,
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
                npz_name = f"{global_offset + i:05d}_{stem}.npz"
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

                img_topk = float("nan")
                if anomaly_map is not None and anomaly_map.size > 0:
                    if anomaly_map.ndim == 2:
                        pixel_scores = anomaly_map
                    else:
                        pixel_scores = anomaly_map.max(axis=-1)
                    img_topk = two_stage_gate_image_score(pixel_scores, args.top_k_fraction)

                save_kw: dict[str, np.ndarray] = {
                    "anomaly_map": anomaly_map.astype(np.float32)
                    if anomaly_map is not None
                    else np.array([], np.float32),
                    "binary_decisions": decisions.astype(np.float32)
                    if decisions is not None
                    else np.array([], np.float32),
                    "gt_mask": mask.astype(np.int32)
                    if mask is not None
                    else np.array([], np.int32),
                    "mesu_index": np.array(int(batch["mesu_index"][i].item()), dtype=np.int64),
                    "cu3s_path": np.array(cu3s_str, dtype=str),
                    "image_score_topk": np.array(img_topk, dtype=np.float32),
                }
                if ascore is not None:
                    save_kw["anomaly_score"] = np.asarray(ascore, dtype=np.float32).reshape(-1)
                if rgb is not None:
                    save_kw["rgb_image"] = rgb.astype(np.float32)
                if m_names.size:
                    save_kw["metric_names"] = m_names
                    save_kw["metric_values"] = m_vals

                np.savez_compressed(npz_path, **save_kw)
                mf.write(
                    json.dumps(
                        {
                            "npz": str(npz_path),
                            "cu3s_path": cu3s_str,
                            "mesu_index": int(batch["mesu_index"][i].item()),
                        }
                    )
                    + "\n"
                )

            global_offset += bsz
            if global_offset % 20 == 0 or global_offset >= n_predict:
                logger.info("Saved frames {}/{}", min(global_offset, n_predict), n_predict)

            if args.limit > 0 and global_offset >= args.limit:
                break

    meta_path.write_text(
        json.dumps(
            {
                "cu3s_file_path": cu3s_str,
                "annotation_json_path": str(ann) if ann else None,
                "n_frames_exported": int(global_offset),
                "n_dataset_frames": int(n_total),
                "pipeline_yaml": str(yaml_path),
                "pipeline_pt": str(pt_path),
                "processing_mode": args.processing_mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Done. NPZ in {} (manifest: {})", out_dir, manifest_path)


if __name__ == "__main__":
    main()
