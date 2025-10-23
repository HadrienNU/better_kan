import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from . import GridBasedFunction


class Splines(GridBasedFunction):
    def __init__(self, in_features, out_features, grid, k=3, fast_version=False, scale=0.1):
        grid.order = k
        self.spline_order = k
        super().__init__(in_features, out_features, grid, fast_version, scale)

    @property
    def n_basis_function(self):
        return self.grid.grid_size + self.spline_order

    def basis(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, grid_size + spline_order, in_features).
        """
        torch._assert(
            x.dim() == 2 and x.size(1) == self.in_features,
            "Input dimension does not match layer size",
        )

        grid: torch.Tensor = self.grid  # (grid_size + 2 * spline_order + 1, in_features)
        x = x.unsqueeze(1)
        bases = ((x >= grid[:-1, :]) & (x < grid[1:, :])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[: -(k + 1), :]) / (grid[k:-1, :] - grid[: -(k + 1), :]) * bases[:, :-1, :]) + ((grid[k + 1 :, :] - x) / (grid[k + 1 :, :] - grid[1:(-k), :]) * bases[:, 1:, :])
        torch._assert(
            bases.size() == (x.size(0), self.n_basis_function, self.in_features),
            f"Basis size not matching expectation {bases.size()}, {(x.size(0), self.n_basis_function, self.in_features)}",
        )
        return bases  # .contiguous()


class GridReLU(GridBasedFunction):
    def __init__(self, in_features, out_features, grid, k=3, fast_version=False, scale=0.1):
        grid.order = k
        self.spline_order = k
        super().__init__(in_features, out_features, grid, fast_version, scale)

    @property
    def n_basis_function(self):
        return self.grid.grid_size + self.spline_order

    def basis(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, grid_size + spline_order, in_features).
        """
        torch._assert(
            x.dim() == 2 and x.size(1) == self.in_features,
            "Input dimension does not match layer size",
        )

        grid: torch.Tensor = self.grid  # (grid_size + 2 * spline_order + 1, in_features)
        x = x.unsqueeze(1)
        x1 = 2 * torch.relu(x - grid[: -(self.spline_order + 1), :]) / (grid[self.spline_order + 1 :, :] - grid[: -(self.spline_order + 1), :])
        x2 = 2 * torch.relu(grid[(self.spline_order + 1) :, :] - x) / (grid[self.spline_order + 1 :, :] - grid[: -(self.spline_order + 1), :])

        bases = x1 * x2
        bases = bases * bases

        torch._assert(
            bases.size() == (x.size(0), self.n_basis_function, self.in_features),
            "Basis size not matching expectation",
        )
        return bases  # .contiguous()
