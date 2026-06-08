#!/usr/bin/env python3
"""Run one-frame inference on the frozen AdaCLIP-band Dinomaly pipeline and save Dinomaly rgb input.

Captures ``rgb_selector.outputs.rgb_image`` (same tensor passed to ``DinomalyDetector`` on the
``rgb_image`` port — before resize / ImageNet normalize inside the detector).

Writes:
  - ``dinomaly_rgb_input_hwc_f32.npy`` — float32 [H, W, 3] on CPU
  - ``dinomaly_rgb_input_meta.txt`` — min / max / mean per channel
  - ``dinomaly_rgb_input_linear_u8.png`` — ``clip(x,0,1) * 255`` -> uint8 RGB
  - ``dinomaly_rgb_input_u8.png`` — same as linear_u8 (explicit 0–255 uint8 file)
  - ``dinomaly_rgb_input_display_minmax_u8.png`` — per-channel min–max stretch to uint8 (easier to view if linear map looks flat)

Example::

    python examples/dump_custom_dinomaly_rgb_input_png.py \\
        --out_dir /tmp/dinomaly_rgb_dump \\
        --mesu_index 27
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def _prepend_cuvis_ai_schemas_with_node_category() -> None:
    """Prefer a full ``cuvis_ai_schemas`` tree (with ``NodeCategory``) over minimal stubs."""
    candidates: list[Path] = []
    env = os.environ.get("CUVIS_AI_SCHEMAS_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.extend(sorted(Path.home().glob(".cache/uv/archive-v0/*")))
    for root in candidates:
        init_py = root / "cuvis_ai_schemas" / "enums" / "__init__.py"
        if not init_py.is_file():
            continue
        try:
            text = init_py.read_text(encoding="utf-8")
        except OSError:
            continue
        if "NodeCategory" in text:
            sys.path.insert(0, str(root.resolve()))
            return


_prepend_cuvis_ai_schemas_with_node_category()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from cuvis_ai_core.data.datasets import SingleCu3sDataModule  # noqa: E402
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline  # noqa: E402
from cuvis_ai_core.utils.node_registry import NodeRegistry  # noqa: E402
from cuvis_ai_schemas.enums import ExecutionStage  # noqa: E402
from cuvis_ai_schemas.execution import Context  # noqa: E402
from PIL import Image  # noqa: E402


def _rgb_hwc_from_forward(raw: dict[tuple[str, str], Any], *, batch_index: int = 0) -> torch.Tensor:
    """``rgb_selector.outputs.rgb_image`` tensor for one batch element [H, W, 3]."""
    key = ("rgb_selector", "rgb_image")
    if key not in raw:
        keys = [k for k in raw if isinstance(k, tuple) and len(k) == 2 and k[1] == "rgb_image"]
        raise KeyError(f"Missing {key} in pipeline outputs. rgb_image keys: {keys}")
    t = raw[key]
    if not torch.is_tensor(t) or t.ndim != 4 or t.shape[-1] != 3:
        raise TypeError(
            f"Expected rgb_image [B,H,W,3], got {type(t)} shape={getattr(t, 'shape', None)}"
        )
    return t[batch_index].detach().cpu().float()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--yaml_path",
        type=Path,
        default=Path(
            "/mnt/data/cuvis_ai_outputs/dinomaly_rgb_frozen_adaclip_bands_npz_50ep_w0/"
            "trained_models/dinomaly_multifile_rgb_frozen_adaclip_bands.yaml"
        ),
    )
    p.add_argument(
        "--pt_path",
        type=Path,
        default=Path(
            "/mnt/data/cuvis_ai_outputs/dinomaly_rgb_frozen_adaclip_bands_npz_50ep_w0/"
            "trained_models/dinomaly_multifile_rgb_frozen_adaclip_bands.pt"
        ),
    )
    p.add_argument(
        "--plugins_manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "plugins.yaml",
    )
    p.add_argument(
        "--cu3s_path",
        type=Path,
        default=Path("/mnt/data/lentils_videos/sliding/Auto_003+01.cu3s"),
    )
    p.add_argument(
        "--annotation_json_path",
        type=Path,
        default=Path("/mnt/data/lentils_videos/sliding/Auto_003+01.json"),
    )
    p.add_argument("--mesu_index", type=int, default=27, help="Session measurement index.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    for path in (args.yaml_path, args.pt_path, args.plugins_manifest, args.cu3s_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    registry = NodeRegistry()
    registry.load_plugins(str(args.plugins_manifest))
    pipeline = CuvisPipeline.load_pipeline(
        str(args.yaml_path),
        weights_path=str(args.pt_path),
        device=args.device,
        strict_weight_loading=False,
        node_registry=registry,
    )
    pipeline.torch_layers.eval()

    dm = SingleCu3sDataModule(
        cu3s_file_path=str(args.cu3s_path),
        annotation_json_path=str(args.annotation_json_path),
        train_ids=[],
        val_ids=[],
        test_ids=[],
        predict_ids=[args.mesu_index],
        batch_size=1,
        processing_mode="Reflectance",
        normalize_to_unit=False,
    )
    dm.setup(stage="predict")
    loader = dm.predict_dataloader()
    batch = next(iter(loader))

    def _collate_to_device(b: dict) -> dict:
        out: dict = {}
        for k, v in b.items():
            if torch.is_tensor(v):
                out[k] = v.to(args.device, non_blocking=True)
            elif isinstance(v, np.ndarray):
                t = torch.from_numpy(np.ascontiguousarray(v))
                if t.dtype == torch.uint16:
                    t = t.float()
                out[k] = t.to(args.device, non_blocking=True)
            else:
                out[k] = v
        return out

    batch_dev = _collate_to_device(batch)
    mesu = int(batch_dev["mesu_index"].view(-1)[0].item())

    context = Context(
        stage=ExecutionStage.TEST,
        epoch=0,
        batch_idx=0,
        global_step=mesu,
    )
    raw = pipeline.forward(batch_dev, context=context)
    rgb = _rgb_hwc_from_forward(raw, batch_index=0)
    hwc = rgb.numpy()

    stem = f"mesu_{mesu:05d}"
    npy_path = args.out_dir / f"{stem}_dinomaly_rgb_input_hwc_f32.npy"
    np.save(npy_path, hwc)

    meta_lines = [
        f"mesu_index={mesu}",
        f"shape_hwc={list(hwc.shape)}",
        "dtype=float32",
        f"overall_min={float(hwc.min())} max={float(hwc.max())}",
    ]
    for c, name in enumerate("RGB"):
        ch = hwc[..., c]
        meta_lines.append(
            f"channel_{name}_min={float(ch.min())} max={float(ch.max())} mean={float(ch.mean())}"
        )
    (args.out_dir / f"{stem}_dinomaly_rgb_input_meta.txt").write_text(
        "\n".join(meta_lines) + "\n", encoding="utf-8"
    )

    lin = np.clip(hwc, 0.0, 1.0)
    u8 = np.round(lin * 255.0).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(u8), mode="RGB").save(
        args.out_dir / f"{stem}_dinomaly_rgb_input_linear_u8.png"
    )
    Image.fromarray(np.ascontiguousarray(u8), mode="RGB").save(
        args.out_dir / f"{stem}_dinomaly_rgb_input_u8.png"
    )

    disp = np.zeros_like(hwc, dtype=np.float32)
    for c in range(3):
        ch = hwc[..., c].astype(np.float32)
        lo, hi = float(ch.min()), float(ch.max())
        if hi - lo < 1e-8:
            disp[..., c] = 0.0
        else:
            disp[..., c] = (ch - lo) / (hi - lo)
    disp_u8 = np.round(np.clip(disp, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(disp_u8), mode="RGB").save(
        args.out_dir / f"{stem}_dinomaly_rgb_input_display_minmax_u8.png"
    )

    print("Wrote:")
    for f in sorted(args.out_dir.glob(f"{stem}_*")):
        print(" ", f)


if __name__ == "__main__":
    main()
