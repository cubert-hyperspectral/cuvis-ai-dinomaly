"""Hyperspectral band selectors used by the Dinomaly plugin.

This file currently provides :class:`FixedHyperspectralSelector` — a generalisation
of :class:`cuvis_ai.node.channel_selector.FixedWavelengthSelector` that accepts any
number of target wavelengths (the upstream version is hard-typed to a 3-tuple).
It is intentionally a thin, no-normalize selector: experiments in the bedding pilot
(May 2026) showed that the upstream selector's per-frame min-max normalize compresses
small-anomaly Dice — see :func:`eval_bedding_rgb.optimal_f1_dice(use_raw=True)`.

Why a separate node instead of generalising the upstream selector
-----------------------------------------------------------------
Keeping the change inside the plugin (a) avoids a cross-repo PR blocker for the
6-channel bedding-all6 pilot and (b) bakes in the no-normalize default the bedding
A/B already validated. If a future selector needs the running-bounds / statistical
norm-mode behaviour, the upstream selector remains the right choice.
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
