"""Antialiased bilinear resize as a fixed linear operator, for ONNX export.

Torchvision's antialiased bilinear resize (``torchvision.transforms.v2.Resize``,
the default the node's preprocessing uses) lowers to an ONNX ``Resize`` node with
antialiasing enabled. TensorRT refuses that node ("Antialiasing is not supported
currently"). At a fixed input and output size the resize is linear and separable,
so it is exactly ``A_h @ X @ A_w^T`` per channel, with ``A_h`` (H_out x H_in) and
``A_w`` (W_out x W_in) built once from the real resize. Applying the two matrices
with ``einsum`` lowers to ``MatMul``/``Gemm`` nodes, which TensorRT accepts.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import resize


def _axis_matrix(size_in: int, size_out: int, other_size: int, *, resize_h: bool) -> Tensor:
    """The ``size_out x size_in`` matrix of the antialiased resize along one axis.

    Built by feeding the identity (one unit vector per input position) through
    torchvision's own resize and reading the result off as the matrix's columns.
    The probe image's other axis is set to ``other_size`` (a real image size, not
    1): torchvision's antialiased kernel treats a size-1 axis as a different
    boundary case and gives a resize matrix that does not match a real image
    (measured ~0.9 max abs difference against the true two-axis resize; at a
    real probe size the two agree to float32 rounding, ~3e-5 at a 448->392
    downscale). The probe is built and resized once in float64 for a stable
    matrix, then cast to the working dtype.
    """
    eye = torch.eye(size_in, dtype=torch.float64)
    if resize_h:
        probe = eye.reshape(size_in, 1, size_in, 1).expand(size_in, 1, size_in, other_size)
        out = resize(
            probe.contiguous(),
            [size_out, other_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        matrix = out[:, 0, :, 0].T
    else:
        probe = eye.reshape(size_in, 1, 1, size_in).expand(size_in, 1, other_size, size_in)
        out = resize(
            probe.contiguous(),
            [other_size, size_out],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        matrix = out[:, 0, 0, :].T
    return matrix.contiguous()


class AntialiasResizeMatrix(nn.Module):
    """Antialiased bilinear resize from a fixed ``(h_in, w_in)`` to a fixed
    ``(h_out, w_out)``, as two buffers applied with matrix multiplication.

    The matrices are built lazily on the first eager call (outside tracing) from
    the actual input size, so the module adapts to whatever size it first sees;
    every later call must carry that same size, matching the exporter's
    fixed-shape contract. Calling it under ``torch.compiler.is_compiling()``
    before the matrices exist raises, since a traced graph cannot run the
    (control-flow-heavy) matrix construction itself.
    """

    def __init__(self, size_out: tuple[int, int]) -> None:
        super().__init__()
        self.size_out = (int(size_out[0]), int(size_out[1]))
        self._built = False
        self._size_in: tuple[int, int] | None = None

    def _build(self, size_in: tuple[int, int], dtype: torch.dtype, device: torch.device) -> None:
        h_in, w_in = size_in
        h_out, w_out = self.size_out
        a_h = _axis_matrix(h_in, h_out, w_in, resize_h=True).to(dtype=dtype, device=device)
        a_w = _axis_matrix(w_in, w_out, h_in, resize_h=False).to(dtype=dtype, device=device)
        self.register_buffer("a_h", a_h, persistent=False)
        self.register_buffer("a_w", a_w, persistent=False)
        self._size_in = size_in
        self._built = True

    def forward(self, x: Tensor) -> Tensor:
        """Resize ``x`` (``[B, C, H, W]``) to ``size_out``."""
        size_in = (int(x.shape[-2]), int(x.shape[-1]))
        if not self._built:
            if torch.compiler.is_compiling():
                raise RuntimeError(
                    "AntialiasResizeMatrix: the resize matrices are not built yet. Run one "
                    "eager forward (outside tracing) on a real input before exporting, so "
                    "the matrices exist when the module is traced."
                )
            self._build(size_in, x.dtype, x.device)
        elif size_in != self._size_in:
            raise RuntimeError(
                f"AntialiasResizeMatrix: built for input size {self._size_in}, called with "
                f"{size_in}. The module is fixed to one input size once built."
            )
        return torch.einsum("oh,bchw,pw->bcop", self.a_h, x, self.a_w)
