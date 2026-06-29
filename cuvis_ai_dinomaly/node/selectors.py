"""Hyperspectral band selectors used by the Dinomaly plugin.

This file provides :class:`FixedHyperspectralSelector` — a thin, no-normalize
selector that picks the nearest band for each of ``n`` target wavelengths and
stacks them in order (``[B, H, W, n]``).

Upstream parity
---------------
:class:`cuvis_ai.node.channel_selector.FixedWavelengthSelector` was generalised to
``n`` channels in cuvis-ai#39 (released in cuvis-ai 0.9+) and now produces
**bit-for-bit identical** output to this class for the no-normalize path (same
``nearest-band-per-target → torch.stack`` logic; verified, max abs diff 0.0).
Migration note: upstream requires ``norm_mode="per_frame"`` for ``n != 3`` (its
default ``"running"`` raises, since the running/statistical paths assume 3-element
buffers). Originally this plugin copy existed because the upstream selector was
hard-typed to a 3-tuple.

Why this copy is retained (not deleted post-#39)
------------------------------------------------
(a) The plugin must not depend on the high-level ``cuvis-ai`` package — importing it
    eagerly loads the proprietary Cuvis SDK (see pyproject), so a plugin-registered
    selector node cannot pull ``FixedWavelengthSelector`` from there.
(b) Already-saved pipelines — including the published HF model
    ``cubert-gmbh/dinomaly-bedding-all6`` — reference this class by import path;
    deleting it would break loading them without a pipeline re-save + re-upload.
Fully switching the *bedding workflow* (notebooks + cookbook training script, which
already use high-level ``cuvis-ai`` nodes) to the upstream selector is folded into
the core-0.10 migration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec


class FixedHyperspectralSelector(Node):
    """Select a fixed set of bands from a hyperspectral cube and stack them in order.

    For each target wavelength, picks the nearest band in the input ``wavelengths``
    array and stacks the selected bands in target order, producing
    ``[B, H, W, len(target_wavelengths)]``. No per-frame normalisation is applied —
    the upstream :class:`MinMaxNormalizer` (if present) already provides the running
    [0, 1] range the downstream model expects.

    Parameters
    ----------
    target_wavelengths
        Target wavelengths in nanometers, in the order the downstream model expects
        them stacked. For Dinomaly bedding-all6 this is
        ``(625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)`` — descending λ within each
        triplet so SWIR-longest lands in the patch-embed R-slot duplicate
        (see ``DinomalyDetector`` docstring).
    normalize_output
        If ``True``, divide the output by its per-frame max. Default ``False`` to keep
        raw reflectance for downstream raw-score Dice computation. Kept as a kwarg for
        API parity with :class:`cuvis_ai.node.channel_selector.FixedWavelengthSelector`.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.DIM_REDUCTION, NodeTag.PREPROCESSING, NodeTag.NUMPY}
    )

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C] in float32",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Wavelength array [C] in nanometers",
        ),
    }

    OUTPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Stacked selected bands [B, H, W, len(target_wavelengths)]. "
            "Port name kept as 'rgb_image' for pipeline-graph compatibility "
            "with DinomalyDetector and other downstream consumers — the "
            "channel count depends on the target_wavelengths configured.",
        ),
        "band_info": PortSpec(
            dtype=dict,
            shape=(),
            description="Selected band metadata (indices, actual wavelengths, targets).",
        ),
    }

    def __init__(
        self,
        target_wavelengths: tuple[float, ...] = (625.0, 550.0, 450.0),
        normalize_output: bool = False,
        **kwargs: Any,
    ) -> None:
        target_tuple = tuple(float(w) for w in target_wavelengths)
        if len(target_tuple) < 1:
            raise ValueError(
                "FixedHyperspectralSelector: target_wavelengths must contain at least 1 wavelength"
            )
        super().__init__(
            target_wavelengths=target_tuple,
            normalize_output=bool(normalize_output),
            **kwargs,
        )
        self.target_wavelengths = target_tuple
        self.normalize_output = bool(normalize_output)
        # No buffers, no statistical initialization needed.
        self._requires_initial_fit_override = False

    @staticmethod
    def _nearest_band_index(wavelengths: np.ndarray, target_nm: float) -> int:
        return int(np.argmin(np.abs(wavelengths - target_nm)))

    def forward(
        self,
        cube: torch.Tensor,
        wavelengths: Any,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        wavelengths_np = np.asarray(wavelengths, dtype=np.float32).ravel()
        indices = [self._nearest_band_index(wavelengths_np, nm) for nm in self.target_wavelengths]

        bands = [cube[..., idx] for idx in indices]
        out = torch.stack(bands, dim=-1)

        if self.normalize_output:
            mx = out.amax()
            if float(mx) > 1e-8:
                out = out / mx

        band_info = {
            "strategy": "hyperspectral_band_pick",
            "band_indices": indices,
            "band_wavelengths_nm": [float(wavelengths_np[i]) for i in indices],
            "target_wavelengths_nm": list(self.target_wavelengths),
            "normalized_output": self.normalize_output,
        }
        return {"rgb_image": out, "band_info": band_info}
