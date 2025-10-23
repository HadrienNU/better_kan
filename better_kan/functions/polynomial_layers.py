import torch
from . import BasisFunction


class ChebyshevPolynomial(BasisFunction):
    def __init__(
        self,
        in_features,
        out_features,
        poly_order=3,
        **kwargs,
    ):

        self.poly_order = poly_order
        self.register_buffer("arange", torch.arange(0, self.poly_order + 1, 1))
        super(ChebyshevPolynomial, self).__init__(in_features, out_features, **kwargs)

    @property
    def n_basis_function(self):
        return self.poly_order + 1
    
    def collocations_points(self):
        return ??

    def basis(self, x: torch.Tensor):
        """
        Compute the Chebyshev bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Chebyshev bases tensor of shape (batch_size, poly_order, in_features).
        """
        torch._assert(
            x.dim() == 2 and x.size(1) == self.in_features,
            "Input dimension does not match layer size",
        )
        x = torch.tanh(
            (x - (self.grid[0, :] + self.grid[-1, :]))
            / (self.grid[-1, :] - self.grid[0, :])
        )  # Rescale into grid
        # View and repeat input degree + 1 times
        x = x.view((-1, self.in_features, 1)).expand(
            -1, -1, self.poly_order + 1
        )  # shape = (batch_size, in_features, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos().transpose(1, 2)
        return x

    def get_subset(self, in_id, out_id, new_grid_size=None):
        """
        get a smaller basis from a larger basis (used for pruning)

        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
        """

        newbasis = ChebyshevPolynomial(
            len(in_id),
            len(out_id),
            poly_order=self.poly_order,
        )
        newbasis.set_from_another_basis(self, in_id, out_id)
        return newbasis


class HermitePolynomial(BasisFunction):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        mask=None,
        poly_order=3,
        scale_base=1.0,
        scale_basis=0.1,
        base_activation=torch.nn.SiLU,
        grid_alpha=0.02,
        grid_range=[-1, 1],
        sb_trainable=True,
        sbasis_trainable=False,
        bias_trainable=True,
        fast_version=False,
        auto_grid_update=False,
        auto_grid_allow_outside_points=0.5,
        pooling_op="sum",
        pooling_args=None,
    ):

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(grid_size) * h + grid_range[0])
            .expand(in_features, -1)
            .transpose(0, 1)
            .contiguous()
        )
        super(HermiteKANLayer, self).__init__(
            in_features,
            out_features,
            grid_size,
            grid,
            base_activation,
            mask,
            scale_base,
            scale_basis,
            grid_alpha,
            False,
            sb_trainable,
            sbasis_trainable,
            bias_trainable,
            fast_version,
            auto_grid_update,
            auto_grid_allow_outside_points,
            grid_size,  # This cancel the trigger on auto_grid_allow_empty_bins,
            pooling_op,
            pooling_args,
        )
        self.poly_order = poly_order

        self.grid_range = grid_range
        self.register_buffer("arange", torch.arange(0, self.poly_order + 1, 1))

        self.reset_parameters()

    @property
    def n_basis_function(self):
        return self.poly_order + 1
    
    def collocations_points(self):
        return ??

    def basis(self, x: torch.Tensor):
        """
        Compute the Hermite bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Hermite bases tensor of shape (batch_size, poly_order, in_features).
        """
        torch._assert(
            x.dim() == 2 and x.size(1) == self.in_features,
            "Input dimension does not match layer size",
        )
        x = torch.tanh(
            (x - (self.grid[0, :] + self.grid[-1, :]))
            / (self.grid[-1, :] - self.grid[0, :])
        )  # Rescale into grid
        hermite = torch.ones(
            x.shape[0], self.poly_order + 1, self.in_features, device=x.device
        )
        if self.poly_order > 0:
            hermite[:, 1, :] = 2 * x
        for i in range(2, self.poly_order + 1):
            hermite[:, i, :] = (
                2 * x * hermite[:, i - 1, :].clone()
                - 2 * (i - 1) * hermite[:, i - 2, :].clone()
            )
        return hermite

