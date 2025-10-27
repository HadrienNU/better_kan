import torch
import torch.nn as nn
import torch.nn.functional as F


from ..utils import assign_parameters


class BasisFunction(nn.Module):
    """
    A base class for functions based on a linear combinaisons of basis
    """

    def __init__(
        self,
        in_features,
        out_features,
        fast_version=False,
    ):
        super(BasisFunction, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        weights = torch.zeros(self.out_features, self.n_basis_function, self.in_features)
        self.weights = torch.nn.Parameter(weights)

        self.set_speed_mode(fast_version)

    # TODO: Allow for more various initialisation of the weights (Box initialisation? Xavier, Kaiming, Sparse,??)
    @torch.no_grad()
    def initialize(self, init_type="uniform", **init_kwargs):
        if init_type == "uniform":
            weights = torch.zeros(self.out_features, self.n_basis_function, self.in_features)
            nn.init.uniform_(weights, **init_kwargs)
        elif init_type == "noise":
            X, w = self.collocations_points()
            noise = (torch.rand(X.shape[0], self.in_features, self.out_features) - 1 / 2) * init_kwargs["scale"] / self.n_basis_function
            weights = self.curve2coeff(X, noise, w)  # Ici prendre des points de collocation plutôt

        assign_parameters(self, "weights", weights)

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

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        if self.fast_mode:
            output = F.linear(
                self.basis(x).reshape(x.size(0), -1),
                self.weights.view(self.out_features, -1),
            )
            return output.view(*original_shape[:-1], self.out_features)
        else:
            output = torch.einsum("xbi,obi->xoi", self.basis(x), self.weights)
            return output.view(*original_shape[:-1], self.out_features, self.in_features)  # Here is the bottleneck of performance

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

        raise NotImplementedError

    def set_speed_mode(self, fast=True):
        self.fast_mode = fast

    def get_subset(self, in_id, out_id):
        """
        get a smaller basis from a larger basis (used for pruning)

        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
        """

        # First obtain deep copy

        # Then change evything that is needed

        newbasis = self.__class__(
            len(in_id),
            len(out_id),
            fast_version=self.fast_mode,
        )
        newbasis.set_from_another_basis(self, in_id, out_id)
        return newbasis

    def get_subset(self, in_id=None, out_id=None):
        """
        When pruning, a only subset of input and output are used
        """

        if in_id is None:
            in_id = torch.arange(self.in_features)
        if out_id is None:
            out_id = torch.arange(self.out_features)

        raise NotImplementedError


class GridBasedFunction(BasisFunction):
    """
    Base class for function based on a grid
    """

    def __init__(self, in_features, out_features, grid, fast_version=False):
        super().__init__(in_features, out_features, fast_version)
        self.grid = grid
        self.grid.in_features = in_features
        self.grid._initialize()

    def collocations_points(self):
        return self.grid.collocations_points()

    def update_grid(self, x, grid_size=-1, margin=0.01):
        torch._assert(x.dim() == 2 and x.size(1) == self.in_features, "Input dimension does not match layer size")

        # Save current value of the basis at collocations points before updating the grid
        x_in = self.collocations_points()
        basis_values = self.basis(x_in)
        unreduced_basis_output = torch.sum(basis_values.unsqueeze(1) * self.weights.unsqueeze(0), dim=2)  # (batch, out, in)
        unreduced_basis_output = unreduced_basis_output.transpose(1, 2)  # (batch, in, out)

        self.grid.update(x, grid_size=grid_size, margin=margin)

        # Update weight to compensate for the grid change
        assign_parameters(self, "weights", self.curve2coeff(x_in, unreduced_basis_output))

    def get_subset(self, in_id=None, out_id=None):
        """
        When pruning, a only subset of input and output are used
        """

        if in_id is None:
            in_id = torch.arange(self.in_features)
        if out_id is None:
            out_id = torch.arange(self.out_features)

        super().getsubset(in_id, out_id)
        # Then reduce the size of the new grid
        self.grid.getsubset(in_id, out_id)

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

        super().__init__(in_features, out_features, fast_version=True)
        self.base_activation = base_activation()

    @property
    def n_basis_function(self):
        return 1

    def collocations_points(self):  # Return a number of collocations points along each input dimensions. When there is an input grid, use the grid
        raise NotImplementedError

    def basis(self, x):
        return self.base_activation(x).unsqueeze(1)

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        if self.fast_mode:
            output = F.linear(self.base_activation(x), self.weights.squeeze(1))
            return output.view(*original_shape[:-1], self.out_features)
        else:
            output = self.base_activation(x).unsqueeze(1) * self.weights.squeeze(1).unsqueeze(0)
            return output.view(*original_shape[:-1], self.out_features, self.in_features)  # Here is the bottleneck of performance

        return
