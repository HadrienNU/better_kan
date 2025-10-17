import torch
from . import BasisFunction


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
    phi = torch.ones_like(distances) / (
        torch.ones_like(distances) + distances.pow(2)
    ).pow(0.5)
    return phi


def spline_rbf(distances):
    phi = distances.pow(2) * torch.log(distances + torch.ones_like(distances))
    return phi


def poisson_one_rbf(distances):
    phi = (distances - torch.ones_like(distances)) * torch.exp(-distances)
    return phi


def poisson_two_rbf(distances):
    phi = (
        ((distances - 2 * torch.ones_like(distances)) / 2 * torch.ones_like(distances))
        * distances
        * torch.exp(-distances)
    )
    return phi


def matern32_rbf(distances):
    phi = (torch.ones_like(distances) + 3**0.5 * distances) * torch.exp(
        -(3**0.5) * distances
    )
    return phi


def matern52_rbf(distances):
    phi = (
        torch.ones_like(distances) + 5**0.5 * distances + (5 / 3) * distances.pow(2)
    ) * torch.exp(-(5**0.5) * distances)
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


class RBFKANLayer(BasisFunction):
    def __init__(
        self,
        in_features,
        out_features,
        grid,
        rbf_kernel="gaussian",
        **kwargs,
    ):
        grid = (
            torch.linspace(grid_range[0], grid_range[1], grid_size)
            .expand(in_features, -1)
            .transpose(0, 1)
            .contiguous()
        )

        sigmas = torch.ones(grid_size, in_features)
        self.sigmas = nn.Parameter(sigmas, requires_grad=optimize_grid)
        self.get_sigmas_from_grid()

        super(RBFKANLayer, self).__init__(in_features, out_features, **kwargs)

        # Base functions
        self.rbf_name = rbf_kernel
        self.rbf = rbf_kernels[rbf_kernel]  # Selection of the RBF

    @property
    def n_basis_function(self):
        return self.grid_size + 1

    @torch.no_grad()
    def get_sigmas_from_grid(self):  # Get it from gradient of the sorted grid
        sorted_grid, inds_sort = torch.sort(self.grid, dim=0)
        batch_indices = (
            torch.arange(self.grid.size(1), device=self.sigmas.device)
            .unsqueeze(0)
            .expand_as(inds_sort)
        )
        self.sigmas[inds_sort, batch_indices] = (
            1.2 / torch.gradient(sorted_grid, dim=0)[0]
        )
        return self.sigmas

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, grid_size=-1, margin=0.01):
        torch._assert(
            x.dim() == 2 and x.size(1) == self.in_features,
            "Input dimension does not match layer size",
        )
        batch = x.size(0)
        basis_values = self.basis(x)
        unreduced_basis_output = torch.sum(
            basis_values.unsqueeze(1) * self.weights.unsqueeze(0), dim=2
        )  # (batch, out, in)
        unreduced_basis_output = unreduced_basis_output.transpose(
            1, 2
        )  # (batch, in, out)
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]

        if grid_size > 0:  # Change grid size
            self.grid_size = grid_size
            self.n_basis_function = grid_size

        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size, dtype=x.dtype, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_alpha * grid_uniform + (1 - self.grid_alpha) * grid_adaptive
        assign_parameters(self, "grid", grid)
        self.get_sigmas_from_grid()
        assign_parameters(self, "weights", self.curve2coeff(x, unreduced_basis_output))

    def basis(self, x):
        distances = x.unsqueeze(1) - self.grid.unsqueeze(0)
        return self.rbf(distances * self.sigmas.unsqueeze(0))

    def get_subset(self, in_id, out_id, new_grid_size=None):
        """
        get a smaller KANLayer from a larger KANLayer (used for pruning)

        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons

        Returns:
        --------
            newlayer : KANLayer

        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        """

        newlayer = RBFKANLayer(
            len(in_id),
            len(out_id),
            grid_size=self.grid_size if new_grid_size is None else new_grid_size,
            mask=mask_subset(self, in_id),
            optimize_grid=self.grid.requires_grad,
            scale_base=self.scale_base,
            scale_basis=self.scale_basis,
            base_activation=type(self.base_activation),
            rbf_kernel=self.rbf_name,
            grid_alpha=self.grid_alpha,
            grid_range=self.grid_range,
            sb_trainable=self.base_scaler.requires_grad,
            sbasis_trainable=self.sbasis_trainable,
            bias_trainable=self.bias.requires_grad,
        )
        newlayer.set_from_another_layer(
            self, in_id, out_id
        )  # It copy almost all variables
        newlayer.sigmas.data.copy_(self.sigmas[:, in_id])
        return newlayer
