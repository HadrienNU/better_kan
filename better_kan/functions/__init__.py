#!python3
import torch
import torch.nn as nn


class BasisFunction(torch.nn.Module):
    """
    A base class for functions based on a linear combinaisons of basis
    """

    def __init__(
        self,
        in_features,
        out_features,
        fast_version=False,
        scale=0.1,
    ):
        super(BaseFunction, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        with torch.no_grad():  # "Initialisation
            X = self.collocations_points()
            noise = (torch.rand(X.shape[0], self.in_features, self.out_features) - 1 / 2) * scale / self.n_basis_function
            weights = self.curve2coeff(X, noise)  # Ici prendre des points de collocation plutôt
        self.weights = torch.nn.Parameter(weights)

        self.set_speed_mode(fast_version)

    @property
    def n_basis_function(self):
        raise NotImplementedError

    def collocations_points(self):  # Return a number of collocations points along each input dimensions. When there is an input grid, use the grid
        raise NotImplementedError

    def basis(self, x: torch.Tensor):
        """
        Compute the bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, grid_size + spline_order, in_features).
        """
        torch._assert(x.dim() == 2 and x.size(1) == self.in_features, "Input dimension does not match layer size")

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
        torch._assert(x.dim() == 2 and x.size(1) == self.in_features, "Input dimension does not match layer size")
        torch._assert(y.size() == (x.size(0), self.in_features, self.out_features), " Output tensor sizes does not match expectation")
        A = self.basis(x).permute(2, 0, 1)  # (in_features, batch_size, n_basis_function)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)

        if A.device == "cpu" or True:
            solution = torch.linalg.lstsq(A, B).solution  # (in_features, n_basis_function, out_features)
        else:
            solution = svd_lstsq(A, B)
        result = solution.permute(2, 1, 0)  # (out_features, n_basis_function, in_features)
        torch._assert(
            result.size()
            == (
                self.out_features,
                self.n_basis_function,
                self.in_features,
            ),
            "result sizes does not match expectation",
        )
        return result.contiguous()

    def forward_fast(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        output = F.linear(
            self.basis(x).reshape(x.size(0), -1),
            self.weights.view(self.out_features, -1),
        )

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    def forward_slow(self, x: torch.Tensor):  # TODO: ajouter forwad power, product, fmean
        original_shape = x.shape
        out_acts = self.activations_eval(x)
        # Somes statistics for regularisation and plot
        self.min_vals = torch.min(x, dim=0).values
        self.max_vals = torch.max(x, dim=0).values
        self.l1_norm = torch.mean(torch.abs(out_acts), dim=0) / (self.max_vals - self.min_vals)  # out_dim x in_dim
        if self.pooling_op == "sum":
            output = torch.sum(out_acts, dim=2)
        elif self.pooling_op == "prod":
            output = torch.prod(out_acts, dim=2)
        elif self.pooling_op == "power":
            output = torch.pow(torch.sum(torch.pow(out_acts, self.pooling_power), dim=2), 1 / self.pooling_power)
        elif self.pooling_op == "logsumexp":
            output = torch.logsumexp(out_acts, dim=2)
        elif self.pooling_op == "min":
            output = torch.min(out_acts, dim=2)
        elif self.pooling_op == "max":
            output = torch.max(out_acts, dim=2)
        elif self.pooling_op == "fsum":
            output = self.pooling_args["invf"](torch.sum(self.pooling_args["f"](out_acts), dim=2))
        return (output + self.bias.unsqueeze(0)).view(*original_shape[:-1], self.out_features)

    def activations_eval(self, x: torch.Tensor):
        """
        Return the values of the activations functions evaluated on x
        Slower evaluation but useful for plotting
        """

        torch._assert(x.size(-1) == self.in_features, "Input dimension does not match layer size")
        x = x.view(-1, self.in_features)

        acts_output = torch.einsum("xbi,obi->xoi", self.basis(x), self.weights)  # Here is the bottleneck of performance
        output = base_output + acts_output
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, grid_size=-1, margin=0.01):
        torch._assert(x.dim() == 2 and x.size(1) == self.in_features, "Input dimension does not match layer size")
        batch = x.size(0)
        basis_values = self.basis(x)
        unreduced_basis_output = torch.sum(basis_values.unsqueeze(1) * self.weights.unsqueeze(0), dim=2)  # (batch, out, in)
        unreduced_basis_output = unreduced_basis_output.transpose(1, 2)  # (batch, in, out)
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        if grid_size > 0:
            self.grid_size = grid_size
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size, dtype=torch.int64, device=x.device)]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = torch.arange(self.grid_size, dtype=x.dtype, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin

        grid = self.grid_alpha * grid_uniform + (1 - self.grid_alpha) * grid_adaptive
        assign_parameters(self, "grid", grid)

        assign_parameters(self, "weights", self.curve2coeff(x, unreduced_basis_output))

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
        points = torch.linspace(0, old_grid.size(0) - 1.0, self.grid.size(0), dtype=x_sorted.dtype, device=self.grid.device)
        indices = torch.floor(points)
        floating_part = (points - indices).unsqueeze(1)
        indices = indices.to(torch.int64)
        # The next two line deal with the special case of the end point
        floating_part[indices == old_grid.size(0) - 1, :] += 1.0
        indices[indices == old_grid.size(0) - 1] -= 1
        new_grid = x_sorted[indices] * (1.0 - floating_part) + x_sorted[indices + 1] * floating_part

        assign_parameters(self, "grid", new_grid[:, in_id])

        # After that the weight need to be actualized
        basis_values = parent.basis(new_grid)
        unreduced_basis_output = torch.sum(basis_values.unsqueeze(1) * parent.weights.unsqueeze(0), dim=2)  # (batch, out, in)
        unreduced_basis_output = unreduced_basis_output.transpose(1, 2)[:, in_id][:, :, out_id]

        assign_parameters(self, "weights", self.curve2coeff(self.grid, unreduced_basis_output))

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
        if out_id is None:
            out_id = torch.arange(self.out_features)

        self.bias.data.copy_(parent.bias[out_id])
        assign_parameters(self, "base_scaler", parent.base_scaler[out_id][:, in_id])
        self.initialize_grid_from_parent(parent, in_id, out_id)  # This will set grid and weights for the basis

        return self

    def set_speed_mode(self, fast=True):
        if fast:
            self.forward = self.forward_fast
            if hasattr(self, "l1_norm"):
                del self.l1_norm
            if self.pooling_op != "sum":
                print(f"Fast mode is incompatible with pooling operation {self.pooling_op}")
        else:
            self.forward = self.forward_slow


class ActivationFunction(BasisFunction):
    """
    Use an unique activation function
    """

    def __init__(
        self,
        in_features,
        out_features,
        base_activation=torch.nn.SiLU,
        fast_version=False,
        scale=0.1,
    ):

        super().__init__(in_features, out_features, scale=scale, fast_version=True)

    @property
    def n_basis_function(self):
        return 1

    def collocations_points(self):  # Return a number of collocations points along each input dimensions. When there is an input grid, use the grid
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        return F.linear(self.base_activation(x), self.weights.squeeze(1))
