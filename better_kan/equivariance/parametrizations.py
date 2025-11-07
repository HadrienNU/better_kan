"""
Set of modules and function to deal with permutation invariance of the input
"""

import torch
import torch.nn as nn

from representation import *


def parametrize_kan_equivariance(model, equiv_list):
    """
    Set the parametrization for the entire network
    """


def parametrize_layer_equivariance(layer, equiv):
    """
    Take a KANLayer and parametrize what is needed.
    """


class EquivariantVector(nn.Module):
    """
    Equivariant module for shape (n_features,) with the dimension being equivariant
    """

    def __init__(self, rep, dtype=torch.float) -> None:
        super().__init__()
        self.rep = rep
        self.register_buffer("basis", torch.tensor(rep.equivariant_basis(), dtype=dtype))

    def forward(self, X):
        return self.basis @ X

    def right_inverse(self, x):
        return self.basis.T @ x


class EquivariantGrid(nn.Module):
    """
    Equivariant module for shape (..., in_features) where the last dimension is equivariant
    """

    def __init__(self, rep, dtype=torch.float) -> None:
        super().__init__()
        self.rep = rep
        self.out_features = rep.size()
        self.register_buffer("basis", torch.tensor(rep.equivariant_basis(), dtype=dtype))

    def forward(self, X):
        return X @ self.basis.T

    def right_inverse(self, x):
        return x @ self.basis


class EquivariantMatrix(nn.Module):
    """
    Equivariant module for shape (out_features,in_features) where the first and last dimension are equivariant and
    """

    def __init__(self, rep_in, rep_out, dtype=torch.float):
        super().__init__()
        self.rep = rep_out * rep_in.T
        self.register_buffer("basis", torch.tensor(self.rep.equivariant_basis(), dtype=dtype))
        self.out_features = rep_out.size()
        self.in_features = rep_in.size()

    def forward(self, X):
        return (self.basis @ X).reshape(self.out_features, self.in_features)

    def right_inverse(self, x):
        return self.basis.T @ x.reshape(-1)


class EquivariantBasisWeight(nn.Module):
    """
    Equivariant module for shape (out_features,..., in_features) where the first and last dimension are equivariant and
    """

    def __init__(self, rep_in, rep_out, dtype=torch.float):
        super().__init__()
        self.rep = rep_out * rep_in.T
        self.register_buffer("basis", torch.tensor(self.rep.equivariant_basis(), dtype=dtype))
        self.out_features = rep_out.size()
        self.in_features = rep_in.size()

    def forward(self, X):
        X = (X @ self.basis.T).reshape(*X.shape[:-1], self.out_features, self.in_features)
        return X.permute(X.ndim - 2, *torch.arange(0, X.ndim - 2), X.ndim - 1)

    def right_inverse(self, x):
        x = x.permute(*torch.arange(1, x.ndim - 1), 0, x.ndim - 1)  # Set the first dimension to be the
        return x.flatten(start_dim=x.ndim - 2) @ self.basis


if __name__ == "__main__":
    import torch
    from representation import V
    from groups import S
    from utils import draw_matrix_parametrizations
    import matplotlib.pyplot as plt

    G = S(5)
    G2 = S(5)

    rep = V(G)
    rep2 = V(G2)

    # vect = EquivariantGrid(rep)
    # print(vect.basis.shape)

    # a = torch.rand(3, vect.rep.size())
    # print(a.shape)
    # w = vect.right_inverse(a)
    # print("Weights", w.shape)
    # print(vect(w), vect(w).shape)
    # draw_matrix_parametrizations(vect, a)
    mat = EquivariantMatrix(rep, rep2)
    print("Basis", mat.basis.shape)
    # print(mat.basis.reshape(rep.size(), rep2.size(), 2)[:, :, 0])
    # print(mat.basis.reshape(rep.size(), rep2.size(), 2)[:, :, 1])
    a = torch.rand(rep2.size(), rep.size())
    print(a.shape)
    w = mat.right_inverse(a)
    print("Weights", w.shape)
    print(mat(w).shape)
    draw_matrix_parametrizations(mat, a, slice_dims=(0, -1))

    mat = EquivariantBasisWeight(rep, rep2)
    print("Basis", mat.basis.shape)

    a = torch.rand(rep2.size(), 3, 2, rep.size())
    print(a.shape)
    w = mat.right_inverse(a)
    print("Weights", w.shape)
    print(mat(w).shape)
    draw_matrix_parametrizations(mat, a, slice_dims=(0, -1))

    plt.show()
