"""Exact area downscaling for the [H3] Output scale setting (aoleg's fork, aoleg/pruna-P-H3-Image-forge-neo 0.13.1).
Torch only, so CPU tests cover it.

Each output pixel is the mean of the source area it covers, partial pixels weighted by the share they cover
(the same result as cv2's INTER_AREA). It is applied as one weight matrix per axis, so 3/4 and 2/3 are as exact
as 1/2 and 1/4. torch's own "area" mode is adaptive average pooling, which at 3/4 averages uneven windows of one and
two pixels and leaves a faint grid.
"""

from fractions import Fraction

import torch


def scaled_size(size: int, scale) -> int:
    """The side after scaling: the nearest whole pixel, at least 1 (2/3 of 4096 is 2731)."""
    return max(int(round(size * Fraction(scale))), 1)


def area_matrix(size_in: int, size_out: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """(size_out, size_in): row j holds the share of each source pixel in output pixel j, which covers the source
    interval [j * r, (j + 1) * r) with r = size_in / size_out. Every row sums to 1."""
    edges = torch.arange(size_out + 1, dtype=torch.float64) * (size_in / size_out)
    start, stop = edges[:-1, None], edges[1:, None]
    pixel = torch.arange(size_in, dtype=torch.float64)[None, :]
    overlap = (torch.minimum(stop, pixel + 1) - torch.maximum(start, pixel)).clamp(min=0.0)
    return (overlap / (stop - start)).to(dtype=dtype, device=device)


def area_resize(image: torch.Tensor, scale) -> torch.Tensor:
    """(..., H, W) shrunk by `scale` (at most 1) on both axes with an exact area filter."""
    if Fraction(scale) >= 1:
        return image
    height, width = image.shape[-2:]
    rows = area_matrix(height, scaled_size(height, scale), image.dtype, image.device)
    cols = area_matrix(width, scaled_size(width, scale), image.dtype, image.device)
    return rows @ image @ cols.T
