import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


def mask_subset(mask, in_id=None):
    """
    Get new mask for layer subset
    """
    if in_id is not None:
        mask = mask[:, in_id]
    return mask[mask.sum(1).to(dtype=torch.bool), :]  # Remove entry that does not matter


def svd_lstsq(AA, BB, tol=1e-12):
    """
    Workaround to SVD when on CUDA, to allow lstsq on rank-deficient matrices
    Waiting for PR https://github.com/pytorch/pytorch/pull/126652 to be merged
    """
    U, S, Vh = torch.linalg.svd(AA, full_matrices=False)
    Spinv = torch.zeros_like(S)

    Spinv[S > tol * max(AA.shape) * S[0]] = 1 / S[S > tol * max(AA.shape) * S[0]]
    UhBB = U.adjoint() @ BB
    if Spinv.ndim != UhBB.ndim:
        Spinv = Spinv.unsqueeze(-1)
    SpinvUhBB = Spinv * UhBB
    return Vh.adjoint() @ SpinvUhBB


class BasisKANLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size,
        n_basis_function,
        base_activation=torch.nn.SiLU,
        mask=None,  # in_dim x in_features array that contain 0 or 1 to suppress or duplicate parameters of an input
        scale_noise=0.1,
        scale_base=1.0,
        scale_basis=1.0,
        grid_alpha=0.02,
        sb_trainable=True,
        sbasis_trainable=True,
        bias_trainable=True,
        fast_version=False,
        auto_grid_update=False,  # Do we auto update the grid
        auto_grid_allow_outside_points=0.1,
        auto_grid_allow_empty_bins=1,
    ):
        super(BasisKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.n_basis_function = n_basis_function

        if mask is not None:
            self.reduced_in_dim = mask.shape[0]
            assert mask.shape[1] == self.in_features

        else:
            self.reduced_in_dim = self.in_features
            mask = torch.eye(self.in_features)
        self.register_buffer("mask", mask)  #  shape: (self.reduced_in_dim, self.in_features)
        self.register_buffer("inv_mask", torch.linalg.pinv(mask))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_basis = scale_basis

        self.grid_alpha = grid_alpha

        self.base_scaler = torch.nn.Parameter(torch.ones(out_features, self.reduced_in_dim) * self.scale_base, requires_grad=sb_trainable)
        self.basis_scaler = torch.nn.Parameter(torch.ones(out_features, self.reduced_in_dim) * self.scale_basis, requires_grad=sbasis_trainable)

        self.bias = torch.nn.Parameter(torch.zeros(out_features), requires_grad=bias_trainable)

        self.weights = torch.nn.Parameter(torch.Tensor(out_features, n_basis_function, self.reduced_in_dim))

        self.base_activation = base_activation()

        self.set_speed_mode(fast_version)

        # For automatic grid update
        self.auto_grid_update = auto_grid_update
        self.auto_grid_allow_outside_points = auto_grid_allow_outside_points
        self.auto_grid_allow_empty_bins = auto_grid_allow_empty_bins

    def reset_parameters(self):
        # torch.nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)

        self.base_scaler.data = torch.ones(self.out_features, self.reduced_in_dim) * self.scale_base
        self.basis_scaler.data = torch.ones(self.out_features, self.reduced_in_dim) * self.scale_basis

        self.bias.data = torch.zeros(self.out_features)

        # Initialize random splines weight
        with torch.no_grad():
            noise = (torch.rand(self.grid.shape[0], self.reduced_in_dim, self.out_features) - 1 / 2) * self.scale_noise / self.n_basis_function
            noise = torch.einsum("ijk,jl->ilk", noise, self.mask)
            self.scaled_weights = self.curve2coeff(self.grid, noise)

    def basis(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, grid_size + spline_order, in_features).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        raise NotImplementedError

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.basis(x).permute(2, 0, 1)  # (in_features, batch_size, n_basis_function)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        if A.device == "cpu":
            solution = torch.linalg.lstsq(A, B).solution  # (in_features, n_basis_function, out_features)
        else:
            solution = svd_lstsq(A, B)
        result = solution.permute(2, 1, 0)  # (out_features, n_basis_function, in_features)

        assert result.size() == (
            self.out_features,
            self.n_basis_function,
            self.in_features,
        )
        return result.contiguous()

    @property
    def scaled_weights(self):
        return torch.matmul(self.weights * (self.basis_scaler).unsqueeze(1), self.mask)

    @scaled_weights.setter
    def scaled_weights(self, values):
        self.weights.data.copy_(torch.matmul(values, self.inv_mask) / (self.basis_scaler).unsqueeze(1))

    @property
    def scaled_base_weight(self):
        return torch.matmul(self.base_scaler, self.mask)

    @scaled_base_weight.setter
    def scaled_base_weight(self, values):
        self.base_scaler.data.copy_(torch.matmul(values, self.inv_mask))

    @property
    def grid(self):
        return torch.matmul(self._grid, self.mask)

    @grid.setter
    def grid(self, values):
        if self.reduced_in_dim != self.in_features:  # If there is a mask
            new_grid = torch.empty_like(self._grid)  # shape (grid_size, reduced_in_dim)
            # Group together values from the same group
            for n in range(self.reduced_in_dim):
                x_sorted = torch.sort(torch.flatten(values[:, self.mask.to(dtype=torch.bool)[n, :]]), dim=0)[0]
                new_grid[:, n] = x_sorted[torch.linspace(0, x_sorted.shape[0] - 1, self._grid.shape[0], dtype=torch.int64, device=values.device)]
        else:  # Else this is just an identity operation so let skip it
            new_grid = values
        self._grid.copy_(new_grid)

    def forward_fast(self, x: torch.Tensor):
        if self.auto_grid_update:
            if self.trigger_grid_update():
                self.update_grid(x)
        # Fast version that does not allow for regularisation
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.scaled_base_weight)
        basis_output = F.linear(
            self.basis(x).reshape(x.size(0), -1),
            self.scaled_weights.view(self.out_features, -1),
        )
        output = self.bias.unsqueeze(0) + base_output + basis_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    def forward_slow(self, x: torch.Tensor):
        if self.auto_grid_update:
            if self.trigger_grid_update(x):
                self.update_grid(x)
        original_shape = x.shape
        out_acts = self.activations_eval(x)
        # Somes statistics for regularisation and plot
        self.min_vals = torch.min(x, dim=0).values
        self.max_vals = torch.max(x, dim=0).values
        self.l1_norm = torch.mean(torch.abs(out_acts), dim=0) / (self.max_vals - self.min_vals)  # out_dim x in_dim
        output = self.bias.unsqueeze(0) + torch.sum(out_acts, dim=2)
        return output.view(*original_shape[:-1], self.out_features)

    def activations_eval(self, x: torch.Tensor):
        """
        Return the values of the activations functions evaluated on x
        Slower evaluation but useful for plotting
        """

        assert x.size(-1) == self.in_features
        x = x.view(-1, self.in_features)
        base_output = self.base_activation(x).unsqueeze(1) * self.scaled_base_weight.unsqueeze(0)

        acts_output = torch.einsum("xbi,obi->xoi", self.basis(x), self.scaled_weights)  # Here is the bottleneck of performance
        output = base_output + acts_output
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        basis_values = self.basis(x)
        unreduced_basis_output = torch.sum(basis_values.unsqueeze(1) * self.scaled_weights.unsqueeze(0), dim=2)  # (batch, out, in)
        unreduced_basis_output = unreduced_basis_output.transpose(1, 2)  # (batch, in, out)
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size, dtype=torch.int64, device=x.device)]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = torch.arange(self.grid_size, dtype=x.dtype, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin

        grid = self.grid_alpha * grid_uniform + (1 - self.grid_alpha) * grid_adaptive
        self.grid = grid
        self.scaled_weights = self.curve2coeff(x, unreduced_basis_output)

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

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.
        """
        if hasattr(self, "l1_norm"):
            regularization_loss_activation = self.l1_norm.sum()
            p = self.l1_norm / regularization_loss_activation
            regularization_loss_entropy = -torch.sum(p * torch.log(p + 1e-6))  # Regularization to avoid 0 value
            return regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy
        else:
            return torch.tensor(0.0)

    @torch.no_grad()
    def initialize_grid_from_parent(self, parent, in_id=None, out_id=None):
        """
        Get grid from a parent and compute grid and weights
        """
        if in_id is None:
            in_id = torch.arange(self.in_features)
        if out_id is None:
            out_id = torch.arange(self.out_features)
        old_grid = parent.grid

        x_sorted = torch.sort(old_grid, dim=0)[0]
        points = torch.linspace(0, old_grid.size(0) - 1.00001, self.grid.size(0), dtype=x_sorted.dtype, device=self.grid.device)
        indices = torch.floor(points)
        floating_part = (points - indices).unsqueeze(1)
        indices = indices.to(torch.int64)
        new_grid = x_sorted[indices] * (1.0 - floating_part) + x_sorted[indices + 1] * floating_part

        self.grid = new_grid[:, in_id]

        # Et après ça il faut actualiser les poids
        basis_values = parent.basis(new_grid)
        unreduced_basis_output = torch.sum(basis_values.unsqueeze(1) * parent.scaled_weights.unsqueeze(0), dim=2)  # (batch, out, in)
        unreduced_basis_output = unreduced_basis_output.transpose(1, 2)[:, in_id][:, :, out_id]

        self.scaled_weights = self.curve2coeff(self.grid, unreduced_basis_output)

    def set_from_another_layer(self, parent, in_id=None, out_id=None):
        """
        set a smaller KANLayer from a larger KANLayer (used for pruning)


        Args:
        -----
            parent : kan_layer
                An input KANLayer to be set as a subset of this one
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
        >>> kanlayer_small = kanlayer_small.set_from_another_layer(kanlayer_large,[0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        """
        if in_id is None:
            in_id = torch.arange(self.in_features)
        in_id_bool = (parent.mask[:, in_id]).sum(1).to(dtype=torch.bool)
        if out_id is None:
            out_id = torch.arange(self.out_features)

        self.bias.data.copy_(parent.bias[out_id])
        self.base_scaler.data.copy_(parent.base_scaler[out_id][:, in_id_bool])
        self.basis_scaler.data.copy_(parent.basis_scaler[out_id][:, in_id_bool])
        self.initialize_grid_from_parent(parent, in_id, out_id)  # This will set grid and weights for the basis

        return self

    def set_speed_mode(self, fast=True):
        if fast:
            self.forward = self.forward_fast
            if hasattr(self, "l1_norm"):
                del self.l1_norm
        else:
            self.forward = self.forward_slow


class RBFKANLayer(BasisKANLayer):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size,
        base_activation=torch.nn.SiLU,
        mask=None,
        optimize_grid=False,
        grid_range=[-1, 1],  # For initialisation of grid
        scale_noise=0.1,
        scale_base=1.0,
        scale_basis=1.0,
        grid_alpha=0.02,  # Enforce less uniform and more adaptative grid in the update step
        sb_trainable=True,
        sbasis_trainable=True,
        bias_trainable=True,
        fast_version=False,
        auto_grid_update=False,
        auto_grid_allow_outside_points=0.8,
        auto_grid_allow_empty_bins=5,
    ):
        super(RBFKANLayer, self).__init__(
            in_features,
            out_features,
            grid_size,
            grid_size,
            base_activation,
            mask,
            scale_noise,
            scale_base,
            scale_basis,
            grid_alpha,
            sb_trainable,
            sbasis_trainable,
            bias_trainable,
            fast_version,
            auto_grid_update,
            auto_grid_allow_outside_points,
            auto_grid_allow_empty_bins,
        )
        self.optimize_grid = optimize_grid
        # Creating the parameters
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size).expand(self.reduced_in_dim, -1).transpose(0, 1).contiguous()
        self._grid = nn.Parameter(grid, requires_grad=optimize_grid)
        # If optimizing over sigmas
        # self.sigmas = nn.Parameter(torch.Tensor(self.in_features, grid_size), requires_grad=False)
        # Else sigmas are simple derivative of the grid
        sigmas = torch.ones(grid_size, in_features)  # nn.Parameter(torch.Tensor(grid_size,in_dim), requires_grad=False)
        self.register_buffer("sigmas", sigmas)
        self.get_sigmas_from_grid()

        # Base functions
        self.rbf = self.gaussian_rbf  # Selection of the RBF

        # Initialisation parameters
        self.grid_range = grid_range

        self.reset_parameters()

    def reset_parameters(self):
        super(RBFKANLayer, self).reset_parameters()
        # # Reset directly the
        # self._grid = torch.linspace(self.grid_range[0], self.grid_range[1], self.grid_size).expand(self.reduced_in_dim, -1).transpose(0, 1).contiguous()

        # init.trunc_normal_(self.sigmas, mean=self.scale_base, std=1.0)

    def get_sigmas_from_grid(self):
        # C'est gradient mais sur le tensor sorted
        sorted_grid, inds_sort = torch.sort(self.grid, dim=0)
        batch_indices = torch.arange(self.grid.size(1), device=self.sigmas.device).unsqueeze(0).expand_as(inds_sort)
        self.sigmas[inds_sort, batch_indices] = 1.2 / torch.gradient(sorted_grid, dim=0)[0]
        return self.sigmas

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        basis_values = self.basis(x)
        unreduced_basis_output = torch.sum(basis_values.unsqueeze(1) * self.scaled_weights.unsqueeze(0), dim=2)  # (batch, out, in)
        unreduced_basis_output = unreduced_basis_output.transpose(1, 2)  # (batch, in, out)
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size, dtype=torch.int64, device=x.device)]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = torch.arange(self.grid_size, dtype=x.dtype, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin

        grid = self.grid_alpha * grid_uniform + (1 - self.grid_alpha) * grid_adaptive
        self.grid = grid
        self.get_sigmas_from_grid()
        self.scaled_weights = self.curve2coeff(x, unreduced_basis_output)

    def basis(self, x):
        distances = x.unsqueeze(1) - self.grid.unsqueeze(0)
        return self.rbf(distances * self.sigmas.unsqueeze(0))

    def gaussian_rbf(self, distances):
        return torch.exp(-(distances.pow(2)))

    def quadratic_rbf(self, distances):
        phi = distances.pow(2)
        return phi

    def inverse_quadratic_rbf(self, distances):
        phi = torch.ones_like(distances) / (torch.ones_like(distances) + distances.pow(2))
        return phi

    def multiquadric_rbf(self, distances):
        phi = (torch.ones_like(distances) + distances.pow(2)).pow(0.5)
        return phi

    def inverse_multiquadric_rbf(self, distances):
        phi = torch.ones_like(distances) / (torch.ones_like(distances) + distances.pow(2)).pow(0.5)
        return phi

    def spline_rbf(self, distances):
        phi = distances.pow(2) * torch.log(distances + torch.ones_like(distances))
        return phi

    def poisson_one_rbf(self, distances):
        phi = (distances - torch.ones_like(distances)) * torch.exp(-distances)
        return phi

    def poisson_two_rbf(self, distances):
        phi = ((distances - 2 * torch.ones_like(distances)) / 2 * torch.ones_like(distances)) * distances * torch.exp(-distances)
        return phi

    def matern32_rbf(self, distances):
        phi = (torch.ones_like(distances) + 3 ** 0.5 * distances) * torch.exp(-(3 ** 0.5) * distances)
        return phi

    def matern52_rbf(self, distances):
        phi = (torch.ones_like(distances) + 5 ** 0.5 * distances + (5 / 3) * distances.pow(2)) * torch.exp(-(5 ** 0.5) * distances)
        return phi

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
            mask=mask_subset(self.mask, in_id),
            optimize_grid=self.optimize_grid,
            scale_noise=self.scale_noise,
            scale_base=self.scale_base,
            scale_basis=self.scale_basis,
            base_activation=type(self.base_activation),
            grid_alpha=self.grid_alpha,
            grid_range=self.grid_range,
            sb_trainable=self.base_scaler.requires_grad,
            sbasis_trainable=self.basis_scaler.requires_grad,
            bias_trainable=self.bias.requires_grad,
        )
        newlayer.set_from_another_layer(self, in_id, out_id)  # It copy almost all variables
        newlayer.sigmas.data.copy_(self.sigmas[:, in_id])
        return newlayer


class SplinesKANLayer(BasisKANLayer):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        mask=None,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_basis=1.0,
        base_activation=torch.nn.SiLU,
        grid_alpha=0.02,  # !!! Ensure grid_alpha is small enough when running on CUDA to avoid singular basis matrices
        grid_range=[-1, 1],
        sb_trainable=True,
        sbasis_trainable=True,
        bias_trainable=True,
        fast_version=False,
        auto_grid_update=False,
        auto_grid_allow_outside_points=0.01,
        auto_grid_allow_empty_bins=1,
    ):
        super(SplinesKANLayer, self).__init__(
            in_features,
            out_features,
            grid_size,
            grid_size + spline_order,
            base_activation,
            mask,
            scale_noise,
            scale_base,
            scale_basis,
            grid_alpha,
            sb_trainable,
            sbasis_trainable,
            bias_trainable,
            fast_version,
            auto_grid_update,
            auto_grid_allow_outside_points,
            2 * self.spline_order - 1 + auto_grid_allow_empty_bins,
        )
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(self.reduced_in_dim, -1).transpose(0, 1).contiguous()
        self.register_buffer("_grid", grid)

        self.reset_parameters()

    def basis(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, grid_size + spline_order, in_features).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid  # (grid_size + 2 * spline_order + 1, in_features)
        x = x.unsqueeze(1)
        bases = ((x >= grid[:-1, :]) & (x < grid[1:, :])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[: -(k + 1), :]) / (grid[k:-1, :] - grid[: -(k + 1), :]) * bases[:, :-1, :]) + ((grid[k + 1 :, :] - x) / (grid[k + 1 :, :] - grid[1:(-k), :]) * bases[:, 1:, :])
        assert bases.size() == (
            x.size(0),
            self.n_basis_function,
            self.in_features,
        )
        return bases  # .contiguous()

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        basis_values = self.basis(x)
        unreduced_basis_output = torch.sum(basis_values.unsqueeze(1) * self.scaled_weights.unsqueeze(0), dim=2)  # (batch, out, in)
        unreduced_basis_output = unreduced_basis_output.transpose(1, 2)  # (batch, in, out)
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = torch.arange(self.grid_size + 1, dtype=x.dtype, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin

        grid = self.grid_alpha * grid_uniform + (1 - self.grid_alpha) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid = grid
        self.scaled_weights = self.curve2coeff(x, unreduced_basis_output)

    def get_subset(self, in_id, out_id, new_grid_size=None):  # TODO, en faire une par
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

        newlayer = SplinesKANLayer(
            len(in_id),
            len(out_id),
            grid_size=self.grid_size if new_grid_size is None else new_grid_size,
            mask=mask_subset(self.mask, in_id),
            spline_order=self.spline_order,
            scale_noise=self.scale_noise,
            scale_base=self.scale_base,
            scale_basis=self.scale_basis,
            base_activation=type(self.base_activation),
            grid_alpha=self.grid_alpha,
            grid_range=[-1, 1],  # We don't care since the grid is going to be redefined
            sb_trainable=self.base_scaler.requires_grad,
            sbasis_trainable=self.basis_scaler.requires_grad,
            bias_trainable=self.bias.requires_grad,
        )
        newlayer.set_from_another_layer(self, in_id, out_id)
        return newlayer


class ChebyshevKANLayer(BasisKANLayer):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        mask=None,
        chebyshev_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_basis=1.0,
        base_activation=torch.nn.SiLU,
        grid_alpha=0.02,
        grid_range=[-1, 1],
        sb_trainable=True,
        sbasis_trainable=True,
        bias_trainable=True,
        fast_version=False,
        auto_grid_update=False,
        auto_grid_allow_outside_points=0.5,
    ):
        super(ChebyshevKANLayer, self).__init__(
            in_features,
            out_features,
            grid_size,
            chebyshev_order + 1,
            base_activation,
            mask,
            scale_noise,
            scale_base,
            scale_basis,
            grid_alpha,
            sb_trainable,
            sbasis_trainable,
            bias_trainable,
            fast_version,
            auto_grid_update,
            auto_grid_allow_outside_points,
            grid_size,  # This cancel the trigger on auto_grid_allow_empty_bins
        )
        self.chebyshev_order = chebyshev_order

        self.grid_range = grid_range

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(grid_size) * h + grid_range[0]).expand(self.reduced_in_dim, -1).transpose(0, 1).contiguous()
        self.register_buffer("_grid", grid)

        self.register_buffer("arange", torch.arange(0, self.chebyshev_order + 1, 1))

        self.reset_parameters()

    def basis(self, x: torch.Tensor):
        """
        Compute the Chebyshev bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Chebyshev bases tensor of shape (batch_size, chebyshev_order, in_features).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        x = torch.tanh((x - (self.grid[0, :] + self.grid[-1, :])) / (self.grid[-1, :] - self.grid[0, :]))  # Rescale into grid
        # View and repeat input degree + 1 times
        x = x.view((-1, self.in_features, 1)).expand(-1, -1, self.chebyshev_order + 1)  # shape = (batch_size, in_features, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos().transpose(1, 2)
        return x

    def get_subset(self, in_id, out_id, new_grid_size=None):  # TODO, en faire une par
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

        newlayer = ChebyshevKANLayer(
            len(in_id),
            len(out_id),
            grid_size=self.grid_size if new_grid_size is None else new_grid_size,
            mask=mask_subset(self.mask, in_id),
            chebyshev_order=self.chebyshev_order,
            scale_noise=self.scale_noise,
            scale_base=self.scale_base,
            scale_basis=self.scale_basis,
            base_activation=type(self.base_activation),
            grid_alpha=self.grid_alpha,
            grid_range=self.grid_range,
            sb_trainable=self.base_scaler.requires_grad,
            sbasis_trainable=self.basis_scaler.requires_grad,
            bias_trainable=self.bias.requires_grad,
        )
        newlayer.set_from_another_layer(self, in_id, out_id)
        return newlayer
