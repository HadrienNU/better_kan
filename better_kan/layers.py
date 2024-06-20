import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


class BasisKANLayer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        grid_size,
        n_basis_function,
        base_activation=torch.nn.SiLU,
        mask=None,  # in_dim x input_dim array that contain 0 or 1 to suppress or duplicate parameters of an input
        scale_noise=0.1,
        scale_base=1.0,
        scale_basis=1.0,
        grid_alpha=0.02,
    ):
        super(BasisKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.n_basis_function = n_basis_function

        if mask is not None:
            self.reduced_in_dim = mask.shape[0]
            assert mask.shape[1] == self.input_dim

        else:
            self.reduced_in_dim = self.input_dim
            mask = torch.eye(self.input_dim)
        self.register_buffer("mask", mask)
        self.register_buffer("inv_mask", torch.linalg.pinv(mask))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_basis = scale_basis

        self.grid_alpha = grid_alpha

        self.base_scaler = torch.nn.Parameter(torch.ones(output_dim, self.reduced_in_dim) * self.scale_base)
        self.basis_scaler = torch.nn.Parameter(torch.ones(output_dim, self.reduced_in_dim) * self.scale_basis)

        self.bias = torch.nn.Parameter(torch.zeros(output_dim))

        self.weights = torch.nn.Parameter(torch.Tensor(output_dim, n_basis_function, self.reduced_in_dim))

        self.base_activation = base_activation()

    def reset_parameters(self):
        # torch.nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)

        self.base_scaler.data = torch.ones(self.output_dim, self.reduced_in_dim) * self.scale_base
        self.basis_scaler.data = torch.ones(self.output_dim, self.reduced_in_dim) * self.scale_basis

        self.bias.data = torch.zeros(self.output_dim)

        # Initialize random splines weight
        with torch.no_grad():
            noise = (torch.rand(self.grid.shape[0], self.reduced_in_dim, self.output_dim) - 1 / 2) * self.scale_noise / self.n_basis_function
            noise = torch.einsum("ijk,jl->ilk", noise, self.mask)
            self.scaled_weights = self.curve2coeff(self.grid, noise)

    def basis(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, grid_size + spline_order, input_dim).
        """
        assert x.dim() == 2 and x.size(1) == self.input_dim

        raise NotImplementedError

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            y (torch.Tensor): Output tensor of shape (batch_size, input_dim, output_dim).

        Returns:
            torch.Tensor: Coefficients tensor of shape (output_dim, input_dim, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.input_dim
        assert y.size() == (x.size(0), self.input_dim, self.output_dim)

        A = self.basis(x).permute(2, 0, 1)  # (input_dim, batch_size, n_basis_function)
        B = y.transpose(0, 1)  # (input_dim, batch_size, output_dim)
        solution = torch.linalg.lstsq(A, B).solution  # (input_dim, n_basis_function, output_dim)  # There is a bug here ?? but why?? remove most permutation??
        result = solution.permute(2, 1, 0)  # (output_dim, n_basis_function, input_dim)

        assert result.size() == (
            self.output_dim,
            self.n_basis_function,
            self.input_dim,
        )
        return result.contiguous()

    # TODO Ici on reshape les array comme il faut pour les masques et autres
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

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        out_acts = self.activations_eval(x)  # C'est un peu plus lent mais ça permet qd même de calculer ce qu'on veut
        # Somes statistics for regularisation and plot
        self.min_vals = torch.min(x, dim=0).values
        self.max_vals = torch.max(x, dim=0).values
        self.l1_norm = torch.mean(torch.abs(out_acts), dim=0) / (self.max_vals - self.min_vals)
        output = self.bias.unsqueeze(0) + torch.sum(out_acts, dim=2)
        return output.view(*original_shape[:-1], self.output_dim)

    def activations_eval(self, x: torch.Tensor):
        """
        Return the values of the activations functions evaluated on x
        Slower evaluation but useful for plotting
        """

        assert x.size(-1) == self.input_dim
        x = x.view(-1, self.input_dim)

        base_output = self.base_activation(x).unsqueeze(1) * self.scaled_base_weight.unsqueeze(0)

        basis_values = self.basis(x)

        acts_output = torch.sum(basis_values.unsqueeze(1) * self.scaled_weights.unsqueeze(0), dim=2)

        output = base_output + acts_output
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.input_dim
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
        self.grid.copy_(grid)
        self.scaled_weights = self.curve2coeff(x, unreduced_basis_output)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.
        """
        regularization_loss_activation = self.l1_norm.sum()
        p = self.l1_norm / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy

    def change_grid(self, new_grid):
        """
        TODO: Allow for changes in num of grid by getting the parameters that fit the best the current values of the activation functions
        C'est comme update grid mais en changeant les dimensions des array internes
        """
        pass

    def set_subset(self, newlayer, in_id, out_id):  # TODO, en faire une par
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
        newlayer = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun)
        # Il faut copier  grid/ mask
        #  weights
        #  base_scaler / basis_scaler
        # Ce qui est nécessaire à la fonction de base
        newlayer.grid.data = self.grid.reshape(self.out_dim, self.in_dim, spb.num + 1)[out_id][:, in_id].reshape(-1, spb.num + 1)
        newlayer.weights.data = self.coef.reshape(self.out_dim, self.in_dim, spb.coef.shape[1])[out_id][:, in_id].reshape(-1, spb.coef.shape[1])
        newlayer.scale_base.data = self.scale_base.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(
            -1,
        )
        newlayer.scale_sp.data = self.scale_sp.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(
            -1,
        )
        newlayer.mask.data = self.mask.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(
            -1,
        )

        newlayer.in_dim = len(in_id)
        newlayer.out_dim = len(out_id)
        return newlayer


class RBFKANLayer(BasisKANLayer):
    def __init__(
        self,
        input_dim,
        output_dim,
        grid_size,
        base_activation=torch.nn.SiLU,
        mask=None,
        optimize_grid=False,
        grid_range=[-1, 1],  # For initialisation of grid
        scale_noise=0.1,
        scale_base=1.0,
        scale_basis=1.0,
        grid_alpha=0.02,  # Enforce less uniform and more adaptative grid in the update step
    ):
        super(RBFKANLayer, self).__init__(input_dim, output_dim, grid_size, grid_size, base_activation, mask, scale_noise, scale_base, scale_basis, grid_alpha)
        self.optimize_grid = optimize_grid
        # Creating the parameters
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(grid_size) * h + grid_range[0]).expand(input_dim, -1).transpose(0, 1).contiguous()
        self.grid = nn.Parameter(grid, requires_grad=optimize_grid)
        # If optimizing over sigmas
        # self.sigmas = nn.Parameter(torch.Tensor(self.input_dim, grid_size), requires_grad=False)
        # Else sigmas are simple derivative of the grid
        sigmas = torch.empty_like(self.grid)  # nn.Parameter(torch.Tensor(grid_size,in_dim), requires_grad=False)
        self.register_buffer("sigmas", sigmas)
        # Choose which parametrization to use
        self.get_activations_params = self.get_sigma_activations_params

        # Base functions
        self.rbf = self.gaussian_rbf  # Selection of the RBF

        # Initialisation parameters
        self.grid_range = grid_range

        self.reset_parameters()

    def reset_parameters(self):
        super(RBFKANLayer, self).reset_parameters()
        # self.grid = torch.linspace(self.grid_range[0], self.grid_range[1], self.grid_size)

        # init.trunc_normal_(self.sigmas, mean=self.scale_base, std=1.0)

    def get_identity_activations_params(self):  # Allow for changing the shape of parameters
        return self.grid, self.sigmas

    def get_sigma_activations_params(self):
        # C'est gradient mais sur le tensor sorted
        sorted_grid, inds_sort = torch.sort(self.grid, dim=1)
        batch_indices = torch.arange(self.grid.size(0), device=self.sigmas.device).unsqueeze(-1).expand_as(inds_sort)
        self.sigmas[batch_indices, inds_sort] = 1.2 / torch.gradient(sorted_grid, dim=0)[0]
        return self.grid, self.sigmas

    def basis(self, x):
        grid, sigmas = self.get_activations_params()
        distances = x.unsqueeze(1) - grid.unsqueeze(0)
        return self.rbf(distances * sigmas.unsqueeze(0))

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
        phi = (torch.ones_like(distances) + 3**0.5 * distances) * torch.exp(-(3**0.5) * distances)
        return phi

    def matern52_rbf(self, distances):
        phi = (torch.ones_like(distances) + 5**0.5 * distances + (5 / 3) * distances.pow(2)) * torch.exp(-(5**0.5) * distances)
        return phi

    def get_subset(self, in_id, out_id):  # TODO, en faire une par
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
        newlayer = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun)
        newlayer.set_subset(in_id, out_id)  # Ca créer tous les trucs de base
        # Il faut copier  grid/ mask
        # Ce qui est nécessaire à la fonction de base
        # On copie ici les trucs en plus
        return newlayer


class SplinesKANLayer(BasisKANLayer):
    def __init__(
        self,
        input_dim,
        output_dim,
        grid_size=5,
        mask=None,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_basis=1.0,
        base_activation=torch.nn.SiLU,
        grid_alpha=0.02,
        grid_range=[-1, 1],
    ):
        super(SplinesKANLayer, self).__init__(input_dim, output_dim, grid_size, grid_size + spline_order, base_activation, mask, scale_noise, scale_base, scale_basis, grid_alpha)
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(input_dim, -1).transpose(0, 1).contiguous()
        self.register_buffer("grid", grid)

        self.reset_parameters()

    def basis(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, grid_size + spline_order, input_dim).
        """
        assert x.dim() == 2 and x.size(1) == self.input_dim

        grid: torch.Tensor = self.grid  # (grid_size + 2 * spline_order + 1, input_dim)
        x = x.unsqueeze(1)
        bases = ((x >= grid[:-1, :]) & (x < grid[1:, :])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[: -(k + 1), :]) / (grid[k:-1, :] - grid[: -(k + 1), :]) * bases[:, :-1, :]) + ((grid[k + 1 :, :] - x) / (grid[k + 1 :, :] - grid[1:(-k), :]) * bases[:, 1:, :])

        assert bases.size() == (
            x.size(0),
            self.n_basis_function,
            self.input_dim,
        )
        return bases.contiguous()

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.input_dim
        batch = x.size(0)

        basis_values = self.basis(x)
        unreduced_basis_output = torch.sum(basis_values.unsqueeze(1) * self.scaled_weights.unsqueeze(0), dim=2)  # (batch, out, in)
        unreduced_basis_output = unreduced_basis_output.transpose(1, 2)  # (batch, in, out)
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin

        grid = self.grid_alpha * grid_uniform + (1 - self.grid_alpha) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid)
        self.scaled_weights = self.curve2coeff(x, unreduced_basis_output)

    def get_subset(self, in_id, out_id):  # TODO, en faire une par
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
        newlayer = SplinesKANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun)
        newlayer.set_subset(in_id, out_id)  # Ca créer tous les trucs de base
        # Il faut copier  grid/ mask
        # Ce qui est nécessaire à la fonction de base
        # On copie ici les trucs en plus
        return newlayer
