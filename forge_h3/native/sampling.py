"""MiniMax-H3-Image's sigma table: Forge's own schedules on the checkpoint's grid (ComfyUI comfy/ldm/minimax/sampling.py
MiniMaxH3ImageModelSampling), from aoleg's fork (aoleg/pruna-P-H3-Image-forge-neo, #8).

The checkpoint's output heads are distilled on its own grid (pdd_sigmas). sigma() is linear in the grid index, so Forge's
stock schedules at the grid's step count land on the grid the heads were made for, and at any other count follow the
same curve; the flow shift is unused.

One step past ComfyUI: Simple reads the 1000-entry sigma table at whole timesteps, which puts a 6-step schedule up to
0.001 off the grid. Positions that close to a grid point are snapped onto it, so 6 steps (and 3, 2 or 1) give the
checkpoint's sigmas exactly, as MiniMax's own pipeline uses them; never onto sigma 0, the table's first entry, which a
schedule like Normal would reach before its last step or take the logarithm of.
"""

import torch
from backend.modules.k_prediction import PredictionDiscreteFlow

# in grid intervals: a 6-step Simple schedule is at most 0.004 off a grid point
SNAP = 0.01


class PredictionH3Image(PredictionDiscreteFlow):
    def __init__(self, model_config, grid):
        # kept ascending so index 0 is sigma 0; set before the base class builds its sigma table through sigma()
        object.__setattr__(self, "_grid", _ascending(grid))
        super().__init__(model_config)

    def set_grid(self, grid) -> None:
        """The grid the selected file declares (integration.py reads it from the file header): the table follows it."""
        grid = _ascending(grid)
        if grid != self._grid:
            object.__setattr__(self, "_grid", grid)
            self.set_parameters()

    def _sigma(self, timestep, snap: bool):
        timestep = torch.as_tensor(timestep, dtype=torch.float32)
        grid = torch.tensor(self._grid, dtype=torch.float32, device=timestep.device)
        n = grid.shape[0] - 1
        pos = (timestep / self.multiplier).clamp(0.0, 1.0) * n
        if snap:
            nearest = pos.round()
            pos = torch.where(((pos - nearest).abs() < SNAP) & (nearest > 0), nearest, pos)
        i = pos.floor().long().clamp(max=n - 1)
        return torch.lerp(grid[i], grid[i + 1], pos - i)

    def sigma(self, timestep):
        return self._sigma(timestep, snap=True)

    def percent_to_sigma(self, percent):
        # not snapped: samplers ask for a point just inside an end on purpose (ER SDE and the DPM++ SDE samplers move a
        # first sigma of 1, an infinite log-SNR, to percent_to_sigma(1e-4)); snapped back to 1.0 it made NaN (aoleg's
        # fork, 0.12.1)
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return float(self._sigma(torch.tensor((1.0 - percent) * self.multiplier), snap=False))


def _ascending(grid) -> tuple[float, ...]:
    """The grid decreases from 1 to 0; the table wants it from 0 up."""
    return tuple(float(s) for s in reversed(tuple(grid)))
