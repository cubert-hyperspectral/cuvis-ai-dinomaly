"""Patch-embed weight inflation utility.

Adapts a pretrained ``Conv2d(in_chans=3, ...)`` patch-embedding (e.g. the one used by
DINOv2's ViT encoder, vendored inside anomalib at
``anomalib.models.image.dinomaly.dinov2.layers.patch_embed.PatchEmbed.proj``) to a
new ``in_channels`` count by duplicating the original weights across the input-channel
dimension and rescaling by ``1 / factor`` so the expected pre-activation magnitude is
preserved.

Why this matters
----------------
``Conv2d`` is linear in its input::

    y[c_out, h, w] = sum_{c_in, kh, kw} w[c_out, c_in, kh, kw] * x[c_in, h+kh, w+kw]

If we replicate ``x`` into a 6-channel input ``x' = cat([x, x], dim=1)`` and halve
all weights, the output is identical to the original 3-channel result — proven by the
``test_activation_parity`` unit test. The inflated conv is then used as a *fixed
activation-parity stem*: it stays frozen during training because anomalib runs the
DINOv2 encoder under ``torch.no_grad()``, so ``patch_embed.proj`` never receives a
gradient (see ``DinomalyDetector._freeze_encoder_unfreeze_head``). The extra bands are
folded into the embedding through the duplicated VIS weights rather than learned apart.

This module is intentionally dependency-free (only ``torch.nn``) so the math can be
unit-tested without instantiating anomalib's DinomalyModel or downloading DINOv2 weights.
"""

from __future__ import annotations

import torch
from torch import nn


def inflate_conv2d_input_channels(old: nn.Conv2d, new_in_channels: int) -> nn.Conv2d:
    """Return a new ``Conv2d`` with ``new_in_channels`` inputs, weights duplicate-and-halved.

    Parameters
    ----------
    old
        The original Conv2d (typically the patch-embedding ``proj`` of a pretrained ViT).
        Must have ``old.in_channels`` divide ``new_in_channels`` evenly. Bias, kernel
        size, stride, padding, dilation, groups, and padding_mode are preserved.
    new_in_channels
        Desired input channel count. For Dinomaly bedding-all6 this is 6 (factor 2).

    Returns
    -------
    nn.Conv2d
        A freshly constructed Conv2d with the inflated weights/bias copied in. The
        original ``old`` is not mutated.

    Raises
    ------
    ValueError
        If ``new_in_channels`` is not a positive multiple of ``old.in_channels``.

    Notes
    -----
    Layout: ``new.weight[:, i, :, :] = old.weight[:, i % old.in_channels, :, :] / factor``
    where ``factor = new_in_channels // old.in_channels``. Use this layout in the
    caller's input-channel ordering: for a 3 → 6 inflation, the caller MUST supply
    input channels in the order ``[c0, c1, c2, c0', c1', c2']`` so the SWIR triplet
    pairs with the VIS triplet in the same conv slots.
    """
    if old.in_channels <= 0 or new_in_channels <= 0:
        raise ValueError(
            f"inflate_conv2d_input_channels: positive channel counts required "
            f"(old={old.in_channels}, new={new_in_channels})"
        )
    if new_in_channels % old.in_channels != 0:
        raise ValueError(
            f"inflate_conv2d_input_channels: new_in_channels ({new_in_channels}) must "
            f"be a positive integer multiple of old.in_channels ({old.in_channels})"
        )
    factor = new_in_channels // old.in_channels

    new = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    with torch.no_grad():
        # repeat along the input-channel dim (dim=1), then halve so total activation
        # magnitude on a duplicated input matches the original 3-ch output.
        new.weight.copy_(old.weight.repeat(1, factor, 1, 1) / float(factor))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    new.to(device=old.weight.device, dtype=old.weight.dtype)
    return new
