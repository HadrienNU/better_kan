import torch
import torch.nn as nn
from .base import GridBasedFunction
from .grid import DummyGrid


def gaussian_rbf(distances):
    return torch.exp(-(distances.pow(2)))


def quadratic_rbf(distances):
    phi = distances.pow(2)
    return phi


def inverse_quadratic_rbf(distances):
    phi = torch.ones_like(distances) / (torch.ones_like(distances) + distances.pow(2))
    return phi


def multiquadric_rbf(distances):
    phi = (torch.ones_like(distances) + distances.pow(2)).pow(0.5)
    return phi


def inverse_multiquadric_rbf(distances):
    phi = torch.ones_like(distances) / (torch.ones_like(distances) + distances.pow(2)).pow(0.5)
    return phi


def spline_rbf(distances):
    phi = distances.pow(2) * torch.log(distances + torch.ones_like(distances))
    return phi


def poisson_one_rbf(distances):
    phi = (distances - torch.ones_like(distances)) * torch.exp(-distances)
    return phi


def poisson_two_rbf(distances):
    phi = ((distances - 2 * torch.ones_like(distances)) / 2 * torch.ones_like(distances)) * distances * torch.exp(-distances)
    return phi


def matern32_rbf(distances):
    phi = (torch.ones_like(distances) + 3**0.5 * distances) * torch.exp(-(3**0.5) * distances)
    return phi


def matern52_rbf(distances):
    phi = (torch.ones_like(distances) + 5**0.5 * distances + (5 / 3) * distances.pow(2)) * torch.exp(-(5**0.5) * distances)
    return phi


rbf_kernels = {
    "gaussian": gaussian_rbf,
    "quadratic": quadratic_rbf,
    "inverse quadratic": inverse_quadratic_rbf,
    "multiquadric": multiquadric_rbf,
    "inverse multiquadric": inverse_multiquadric_rbf,
    "spline": spline_rbf,
    "poisson one": poisson_one_rbf,
    "poisson two": poisson_two_rbf,
    "matern32": matern32_rbf,
    "matern52": matern52_rbf,
}


class RBFFunction(GridBasedFunction):
    def __init__(
        self,
        in_features,
        out_features,
        grid,
        rbf_kernel="gaussian",
        optimize_sigmas=False,
        **kwargs,
    ):

        grid.order = 0
        self.grid = DummyGrid(grid)
        super(RBFFunction, self).__init__(in_features, out_features, grid, **kwargs)

        sigmas = torch.ones(self.grid.grid_size, in_features)
        self.sigmas = nn.Parameter(sigmas, requires_grad=optimize_sigmas)
        # self.get_sigmas_from_grid()

        # Base functions
        self.rbf_name = rbf_kernel
        self.rbf = rbf_kernels[rbf_kernel]  # Selection of the RBF

    @property
    def n_basis_function(self):
        return self.grid.grid_size

    @torch.no_grad()
    def get_sigmas_from_grid(self):  # Get it from gradient of the sorted grid
        grid = self.grid.grid
        sorted_grid, inds_sort = torch.sort(grid, dim=0)
        batch_indices = torch.arange(grid.size(1), device=self.sigmas.device).unsqueeze(0).expand_as(inds_sort)
        self.sigmas[inds_sort, batch_indices] = 1.2 / torch.gradient(sorted_grid, dim=0)[0]
        return self.sigmas

    def basis(self, x):
        distances = x.unsqueeze(1) - self.grid.grid.unsqueeze(0)
        return self.rbf(distances * self.sigmas.unsqueeze(0))
