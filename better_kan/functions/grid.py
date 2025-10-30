"""
Contain a Grid class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ..utils import assign_parameters


class DummyGrid:
    """
    A Dummy grid class to avoid issue with initialisation order with nn.Module,
    only hold some grid attribute
    """

    def __init__(self, grid):
        self.grid_size = grid.grid_size
        self.order = grid.order
        self.grid_range = grid.grid_range


class Grid(nn.Module):
    def __init__(self, in_features, size, order=0, grid_range=(-1, 1), grid_alpha=0.02, auto_grid_allow_outside_points=0.1, auto_grid_allow_empty_bins=1):
        super().__init__()

        self.grid_size = size  # Number of point in the grid
        self.order = order  # Define the grid order
        self.grid_range = torch.ones(in_features, 2)
        self.grid_range[:, 0] = grid_range[0]
        self.grid_range[:, 1] = grid_range[1]
        self.grid_alpha = grid_alpha

        self.auto_grid_allow_outside_points = auto_grid_allow_outside_points
        self.auto_grid_allow_empty_bins = auto_grid_allow_empty_bins

        self._initialize()

    def _initialize(self):

        h = (self.grid_range[:, 1] - self.grid_range[:, 0]) / (self.grid_size - 1)
        grid = torch.arange(-self.order, self.grid_size + self.order).unsqueeze(1) * h.unsqueeze(0) + self.grid_range[:, 0].unsqueeze(0)
        self.register_buffer("grid", grid)

    @property
    def n_intervals(self):  # TODO: check the number
        return self.grid_size - 1 + 2 * self.order

    def collocations_points(self, n=None):
        """
        Give a list of collocations points for exact integration against the grid
        n is the polynomial order that give exact integration for Gauss Legendre quadrature
        """

        if n is None:
            n = self.order + 1

        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes_torch = torch.from_numpy(nodes).to(dtype=self.grid.dtype, device=self.grid.device)
        weights_torch = torch.from_numpy(weights).to(dtype=self.grid.dtype, device=self.grid.device)

        nodes_torch = nodes_torch.unsqueeze(1).unsqueeze(2).repeat(1, self.grid_size - 1 + 2 * self.order, self.grid.shape[-1])
        weights_torch = weights_torch.unsqueeze(1).unsqueeze(2).repeat(1, self.grid_size - 1 + 2 * self.order, self.grid.shape[-1])

        # Change of interval
        dh = 0.5 * (self.grid[1:,] - self.grid[:-1, :]).unsqueeze(0)
        center = 0.5 * (self.grid[1:,] + self.grid[:-1, :]).unsqueeze(0)
        print()
        return (center + dh * nodes_torch).view(-1, self.grid.shape[-1]), (dh * weights_torch).view(-1, self.grid.shape[-1])

    # TODO: take care of parametrization
    def update(self, x=None, grid_size=-1, margin=0.01):
        if x is None:  # If no data are provided simply use the previous grid as basis for interpolating to the new size
            if grid_size > 0:
                x_sorted = torch.sort(self.grid, dim=0)[0]
                points = torch.linspace(0, self.grid.size(0) - 1.0, grid_size + 2 * self.order, dtype=self.grid.dtype, device=self.grid.device)
                indices = torch.floor(points)
                floating_part = (points - indices).unsqueeze(1)
                indices = indices.to(torch.int64)
                # The next two line deal with the special case of the end point
                floating_part[indices == self.grid.size(0) - 1, :] += 1.0
                indices[indices == self.grid.size(0) - 1] -= 1
                grid = x_sorted[indices] * (1.0 - floating_part) + x_sorted[indices + 1] * floating_part

                self.grid_size = grid_size
        else:
            torch._assert(x.dim() == 2 and x.size(1) == self.grid.size(1), "Input dimension does not match layer size")

            batch = x.size(0)
            # sort each channel individually to collect data distribution
            x_sorted = torch.sort(x, dim=0)[0]
            if grid_size > 0:
                self.grid_size = grid_size
            grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size, dtype=torch.int64, device=x.device)]

            uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / (self.grid_size - 1)
            grid_uniform = torch.arange(self.grid_size, dtype=x.dtype, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin

            grid = self.grid_alpha * grid_uniform + (1 - self.grid_alpha) * grid_adaptive

            self.grid_range = torch.stack((x_sorted[-1], x_sorted[0]), dim=1)

            if self.order > 0:
                grid = torch.concatenate(
                    [
                        grid[:1] - uniform_step * torch.arange(self.order, 0, -1, device=x.device).unsqueeze(1),
                        grid,
                        grid[-1:] + uniform_step * torch.arange(1, self.order + 1, device=x.device).unsqueeze(1),
                    ],
                    dim=0,
                )
        # TODO: Return a new grid class with the new grid
        # self.grid = grid  # De toute façon ce n'est pas un paramètre
        assign_parameters(self, "grid", grid)

    @torch.no_grad()
    def trigger_grid_update(self, x: torch.Tensor):
        """
        Compute proportion of input that are out of the grid
        That would trigger automatic grid update
        """

        in_bins = ((x.unsqueeze(1) >= self.grid[:-1, :]) & (x.unsqueeze(1) < self.grid[1:, :])).to(x.dtype).sum(dim=0)  # If we want to check if there is a lot of empty bins
        nb_empty_bins = (in_bins == 0).sum(dim=0)

        out_points = torch.logical_or((x >= self.grid[-1, :]), (x < self.grid[0, :])).mean(dim=0, dtype=torch.float64)
        return torch.any(out_points > self.auto_grid_allow_outside_points) or torch.any(nb_empty_bins > self.auto_grid_allow_empty_bins)  # If too many points

    @torch.no_grad()  # TODO: Check the code
    def get_inout_subset(self, in_id=None):
        """
        Get grid from a parent and compute grid and weights
        """
        if in_id is None:
            in_id = torch.arange(self.grid.shape[-1])
        self.grid = self.grid[:, in_id]
        self.grid_range = self.grid_range[in_id, :]
        return self


class ParametrizedGrid(Grid):

    def __init__(self, in_features, size, order=0, grid_range=(-1, 1), grid_alpha=0.02):
        super(ParametrizedGrid, self).__init__(in_features, size, order=order, grid_range=grid_range, grid_alpha=grid_alpha)

    def _initialize(self):
        self.s = nn.Parameter(torch.zeros(self.grid_size - 1, self.grid_range.shape[0]))  # Initialize it as a uniform grid

    @property  # Convert Parameters s  to grid
    def grid(self):
        h = (self.grid_range[:, 1] - self.grid_range[:, 0]).unsqueeze(0)
        grid = self.grid_range[:, 0].unsqueeze(0) + h * torch.cumsum(F.softmax(self.s, dim=0), dim=0)
        # add extra point
        uniform_step = h / (self.grid_size - 1)
        grid = torch.concatenate(
            [
                grid[:1] - uniform_step * torch.arange(self.order + 1, 0, -1).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.order + 1).unsqueeze(1),
            ],
            dim=0,
        )
        return grid

    @grid.setter
    def grid(self, grid):
        """
        From a grid, give the s values
        """
        assert grid.shape == (self.s.shape[0] + 1 + 2 * self.order, self.s.shape[1])
        if self.order > 0:
            grid = grid[self.order : -self.order, :]
        smax = torch.diff(grid, dim=0)
        self.grid_size = smax.shape[0]
        self.s = nn.Parameter(torch.log(smax))
