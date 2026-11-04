"""Mean of the k largest values along the last dimension, without a TopK above a cap.

Anomalib's Dinomaly image score sorts the whole flattened anomaly map descending
and averages the first ``k`` entries (the top one percent). The dynamo exporter
lowers that ``torch.sort(..., descending=True)`` to an ONNX ``TopK`` node whose
``K`` is the full array length, since a sort needs every element ranked. TensorRT
10.13.3 caps ``ITopKLayer`` at ``K = 3840`` ("K exceeds the maximum value
allowed"), so neither the full sort nor a single ``torch.topk(k)`` above that cap
survives the conversion. :func:`bounded_topk_mean` gets the same number (the mean
of the ``k`` largest values) by taking at most 3840 at a time, summing them, and
masking them out before the next round, so the largest ``K`` any ONNX ``TopK``
node sees is 3840.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

#: TensorRT 10.13.3's ``ITopKLayer`` maximum ``K``.
MAX_TOPK = 3840


def bounded_topk_mean(flat: Tensor, k: int, *, max_topk: int = MAX_TOPK) -> Tensor:
    """The mean of the ``k`` largest values of ``flat`` along its last dimension.

    Equivalent to ``torch.topk(flat, k, dim=-1).values.mean(dim=-1)`` (which in
    turn gives the same values, in a different order, as the sorted-and-sliced
    mean anomalib's Dinomaly model computes), but never asks for more than
    ``max_topk`` values in one ``torch.topk`` call. ``k`` must be a plain Python
    int fixed at export time (not a tensor): round count and mask value are
    static, since the exported graph is for a fixed map size.

    Parameters
    ----------
    flat : Tensor
        ``[..., N]``; the top-``k`` is taken independently per leading row.
    k : int
        How many of the largest values to average. Must be at least 1 and at
        most ``flat.shape[-1]``.
    max_topk : int
        The largest ``K`` any single round may ask for. Defaults to
        TensorRT's cap.

    Returns
    -------
    Tensor
        ``[...]``, the mean of the ``k`` largest values per row.
    """
    n = flat.shape[-1]
    if k < 1 or k > n:
        raise ValueError(f"bounded_topk_mean: k must be in [1, {n}], got {k}.")
    rounds = math.ceil(k / max_topk)
    remaining = k
    # The sentinel must be lower than every real value so a masked slot is never
    # picked again; the map minimum minus one guarantees that regardless of sign.
    sentinel = flat.min() - 1
    working = flat
    total = torch.zeros(flat.shape[:-1], dtype=flat.dtype, device=flat.device)
    for round_index in range(rounds):
        take = min(max_topk, remaining)
        values, indices = torch.topk(working, take, dim=-1)
        total = total + values.sum(dim=-1)
        remaining -= take
        if round_index < rounds - 1:
            working = working.scatter(-1, indices, sentinel.expand_as(values))
    return total / k
