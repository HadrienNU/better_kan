"""
Contain a Grid class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..utils import assign_parameters


class Grid(nn.Module):
    def __init__(self, in_features, size, order=0, grid_range=(-1, 1), grid_alpha=0.02):
        super().__init__()

        self.grid_size = size  # Number of point in the grid
        self.order = order  # Define the grid order
        self.grid_range = torch.ones(in_features, 2)
        self.grid_range[:, 0] = grid_range[0]
        self.grid_range[:, 1] = grid_range[1]
        self.grid_alpha = grid_alpha

        self._initialize()

    def _initialize(self):

        h = (self.grid_range[:, 1] - self.grid_range[:, 0]) / (self.grid_size - 1)
        grid = torch.arange(-self.order, self.grid_size + self.order).unsqueeze(1) * h.unsqueeze(0) + self.grid_range[:, 0].unsqueeze(0)
        self.register_buffer("grid", grid)

    def update(self, x, grid_size=-1, margin=0.01):
        batch = x.size(0)
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        if grid_size > 0:
            self.grid_size = grid_size
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size, dtype=torch.int64, device=x.device)]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / (self.grid_size - 1)
        grid_uniform = torch.arange(self.grid_size, dtype=x.dtype, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin

        grid = self.grid_alpha * grid_uniform + (1 - self.grid_alpha) * grid_adaptive

        self.grid_range = torch.concatenate((x_sorted[-1], x_sorted[0]), dim=1)

        if self.order > 0:
            grid = torch.concatenate(
                [
                    grid[:1] - uniform_step * torch.arange(self.order, 0, -1, device=x.device).unsqueeze(1),
                    grid,
                    grid[-1:] + uniform_step * torch.arange(1, self.order + 1, device=x.device).unsqueeze(1),
                ],
                dim=0,
            )
        self.grid = grid  # De toute façon ce n'est pas un paramètre
        # assign_parameters(self, "grid", grid)


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
        grid = torch.concatenate(
            [
                grid[:1] - h * torch.arange(self.order, 0, -1).unsqueeze(1),
                self.grid_range[:, 0].unsqueeze(0),
                grid,
                grid[-1:] + h * torch.arange(1, self.order + 1).unsqueeze(1),
            ],
            dim=0,
        )
        return grid

    @grid.setter
    def grid(self, grid):
        """
        From a grid, give the s values
        """
        if self.order > 0:
            grid = grid[:, self.order : -self.order]
        smax = torch.diff(grid, dim=0)
        self.grid_size = smax.shape[0]
        self.s = nn.Parameter(torch.log(smax))


if __name__ == "__main__":
    grid = Grid(3, 5)

    print(grid.grid)
    print(grid.grid.shape)

    grid = ParametrizedGrid(3, 5)
    print(grid.grid.shape)
    print(grid.grid)
    h = torch.ones((3, 1)) * 0.25
    grid.grid = torch.arange(0, 5).unsqueeze(0) * h
    print(grid.s.shape)
