import torch
from .base import BasisFunction

from numpy.polynomial.legendre import leggauss


class PolynomialFunction(BasisFunction):
    def __init__(
        self,
        in_features,
        out_features,
        poly_order=3,
        **kwargs,
    ):
        self.poly_order = poly_order
        super().__init__(in_features, out_features, **kwargs)
        self.register_buffer("arange", torch.arange(0, self.poly_order + 1, 1))

    @property
    def n_basis_function(self):
        return self.poly_order + 1

    def collocations_points(self):
        nodes, weights = leggauss(self.poly_order + 1)
        nodes_torch = torch.atanh(torch.from_numpy(nodes).to(dtype=self.weights.dtype, device=self.weights.device))
        weights_torch = torch.from_numpy(weights).to(dtype=self.weights.dtype, device=self.weights.device)
        return nodes_torch.unsqueeze(1).expand(-1, self.in_features), weights_torch.unsqueeze(1).expand(-1, self.in_features)


class ChebyshevPolynomial(PolynomialFunction):
    def basis(self, x: torch.Tensor):
        """
        Compute the Chebyshev bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Chebyshev bases tensor of shape (batch_size, poly_order, in_features).
        """
        torch._assert(
            x.dim() == 2,
            "Input dimension does not match layer size",
        )
        x = torch.tanh(x)  # Rescale into [-1,1]
        # View and repeat input degree + 1 times
        x = x.unsqueeze(-1).expand(-1, -1, self.poly_order + 1)  # shape = (batch_size, in_features, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos().transpose(1, 2)
        return x


class HermitePolynomial(PolynomialFunction):

    def basis(self, x: torch.Tensor):
        """
        Compute the Hermite bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Hermite bases tensor of shape (batch_size, poly_order, in_features).
        """
        torch._assert(
            x.dim() == 2,
            "Input dimension does not match layer size",
        )
        x = torch.tanh(x)  # Rescale into grid
        hermite = torch.ones(x.shape[0], self.poly_order + 1, x.size(1), device=x.device)
        if self.poly_order > 0:
            hermite[:, 1, :] = 2 * x
        for i in range(2, self.poly_order + 1):
            hermite[:, i, :] = 2 * x * hermite[:, i - 1, :].clone() - 2 * (i - 1) * hermite[:, i - 2, :].clone()
        return hermite
