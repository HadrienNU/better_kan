import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy.polynomial.hermite import hermgauss


from ..utils import assign_parameters


class BasisFunction(nn.Module):
    """
    A base class for functions based on a linear combinaisons of basis
    """

    def __init__(
        self,
        in_features,
        out_features,
    ):
        super(BasisFunction, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        weights = torch.empty(self.out_features, self.n_basis_function, self.in_features)
        self.weights = torch.nn.Parameter(weights)

        self.reset_parameters(init_type="uniform")

    # TODO: Allow for more various initialisation of the weights (Box initialisation? Xavier, Kaiming, Sparse,??)
    @torch.no_grad()
    def reset_parameters(self, init_type="uniform", **init_kwargs):
        if init_type == "uniform":
            nn.init.uniform_(self.weights, **init_kwargs)
        elif init_type == "noise":
            # Helpers function that return noise of the same shape
            # def noise_fct(X):
            #     return (torch.rand(X.shape[0], self.in_features, self.out_features) - 1 / 2) * init_kwargs["scale"] / self.n_basis_function

            # weights = self.project_on_basis(noise_fct)
            X, w = self.collocations_points()
            noise = (torch.rand(X.shape[0], self.in_features, self.out_features) - 1 / 2) * init_kwargs["scale"] / self.n_basis_function
            weights = self.curve2coeff(X, noise, w)  # Ici prendre des points de collocation plutôt

            assign_parameters(self, "weights", weights)

    @property
    def n_basis_function(self):
        raise NotImplementedError

    def collocations_points(self):  # Return a number of collocations points along each input dimensions. When there is an input grid, use the grid
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        output = F.linear(
            self.basis(x).reshape(x.size(0), -1),
            self.weights.view(self.out_features, -1),
        )
        return output.view(*original_shape[:-1], self.out_features)

    def activations_eval(self, x: torch.Tensor):
        """
        Don't reduce over the input dimension
        """
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        output = torch.einsum("xbi,obi->xoi", self.basis(x), self.weights)
        return output.view(*original_shape[:-1], self.out_features, self.in_features)  # Here is the bottleneck of performance

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

    def project_on_basis(self, fct, method="l2"):
        """
        When fct is a callable function or module; compute the projection of the function on the basis and update the weights

        Args:
            method (str, "l2", "collocation") : allow to change method for the projection
                - "l2" use L2 Galerkin projection
                - "collocation" use least squares projection to a set of collocation points

        Returns;
            weights resulting from the projection
        """
        if method.lower() == "l2":
            raise NotImplementedError
        elif method.lower() == "collocation":
            x, _ = self.collocations_points
            y = fct(x)
            # TODO: check assert
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

    # TODO: Implement the weights
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor, weights=None):
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

    def get_inout_subset(self, in_id=None, out_id=None):
        """
        When pruning, a only subset of input and output are used
        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
        """

        if in_id is None:
            in_id = torch.arange(self.in_features)
        if out_id is None:
            out_id = torch.arange(self.out_features)

        self.in_features = len(in_id)
        self.out_features = len(out_id)
        assign_parameters(self, "weights", self.weights[out_id, :, :][:, :, in_id])
        return self


class GridBasedFunction(BasisFunction):
    """
    Base class for function based on a grid
    """

    def __init__(self, in_features, out_features, grid):
        super().__init__(in_features, out_features)
        self.grid = grid
        self.grid.in_features = in_features
        self.grid._initialize()

    def collocations_points(self):
        return self.grid.collocations_points()

    # TODO: Avoir update grid qui return une nouvelle grille et le curve2coeff etre vraiment la projection d'une base sur l'autre
    @torch.no_grad()
    def update_grid(self, x, grid_size=-1, margin=0.01):
        # Save current value of the basis at collocations points before updating the grid
        if x is not None:
            x_in = x
        else:
            x_in, w = self.collocations_points()
        basis_values = self.basis(x_in)
        unreduced_basis_output = torch.sum(basis_values.unsqueeze(1) * self.weights.unsqueeze(0), dim=2)  # (batch, out, in)
        unreduced_basis_output = unreduced_basis_output.transpose(1, 2)  # (batch, in, out)

        new_grid = self.grid.update(x, grid_size=grid_size, margin=margin)

        # Update weight to compensate for the grid change
        assign_parameters(self, "weights", self.curve2coeff(x_in, unreduced_basis_output))

    def get_inout_subset(self, in_id=None, out_id=None):
        """
        When pruning, a only subset of input and output are used
        """

        if in_id is None:
            in_id = torch.arange(self.in_features)
        if out_id is None:
            out_id = torch.arange(self.out_features)

        super().get_inout_subset(in_id, out_id)
        # Then reduce the size of the new grid
        self.grid.get_inout_subset(in_id)

        return self


class ActivationFunction(BasisFunction):
    """
    Use an unique activation function
    """

    def __init__(
        self,
        in_features,
        out_features,
        base_activation=nn.SiLU,
    ):

        super().__init__(in_features, out_features)
        self.base_activation = base_activation()

    @property
    def n_basis_function(self):
        return 1

    def collocations_points(self):  # Return a number of collocations points along each input dimensions. When there is an input grid, use the grid
        nodes, weights = hermgauss(5)
        nodes_torch = torch.from_numpy(nodes).to(dtype=self.weights.dtype, device=self.weights.device)
        weights_torch = torch.from_numpy(weights).to(dtype=self.weights.dtype, device=self.weights.device)
        return nodes_torch.unsqueeze(1).expand(-1, self.in_features), weights_torch.unsqueeze(1).expand(-1, self.in_features)

    def basis(self, x):
        return self.base_activation(x).unsqueeze(1)

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        output = F.linear(self.base_activation(x), self.weights.squeeze(1))
        return output.view(*original_shape[:-1], self.out_features)

    def activations_eval(self, x):
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        output = self.base_activation(x).unsqueeze(1) * self.weights.squeeze(1).unsqueeze(0)
        return output.view(*original_shape[:-1], self.out_features, self.in_features)  # Here is the bottleneck of performance
