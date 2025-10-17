"""
Set of modules and function to deal with permutation invariance of the input
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import torch.nn.utils.parametrize as parametrize


from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.group_constructs import DirectProduct


def equivariant_permutations_inputs(inputs):
    """
    Return the generators of permutation group for a given input
    inputs is an array that contain the type of the entry, i.e. equivariant entry share the same type

    Return the generators of the symmetry group
    """
    inputs = np.asarray(inputs).ravel()
    nb_inputs = len(inputs)
    types = np.unique(inputs)
    if len(types) == nb_inputs:
        return trivial_group(nb_inputs)
    generators = []
    for t in types:
        if np.sum(inputs == t) > 1:  # Only if more than one entry share the same type
            inds = np.nonzero(inputs == t)[0]
            cycle = Permutation([inds], size=nb_inputs)
            perm = Permutation([inds[:2]], size=nb_inputs)
            generators.append(cycle)
            generators.append(perm)
    end_group = PermutationGroup(generators)
    return [perm.array_form for perm in end_group.generators]


def trivial_group(n):
    """
    Return the trivial group for input of size n
    """

    return PermutationGroup(Permutation([], size=n))


### The following code is borrowed from AutoEquiv https://github.com/mshakerinava/AutoEquiv
### But copied to avoid dependencies to uninstallable librairy


def create_colored_matrix(input_generators, output_generators):
    def dfs(i, j, colors, color_idx, input_generators, output_generators):
        colors[(i, j)] = color_idx
        for k in range(len(input_generators)):
            i_ = input_generators[k][i]
            j_ = output_generators[k][j]
            if (i_, j_) not in colors:
                dfs(i_, j_, colors, color_idx, input_generators, output_generators)

    assert len(input_generators) == len(output_generators)
    assert len(input_generators) > 0
    color_idx = 0
    colors = {}
    for i in range(len(input_generators[0])):
        for j in range(len(output_generators[0])):
            if (i, j) not in colors:
                dfs(i, j, colors, color_idx, input_generators, output_generators)
                color_idx += 1
    return colors


def create_colored_vector(output_generators):
    def dfs(i, colors, color_idx, output_generators):
        colors[i] = color_idx
        for k in range(len(output_generators)):
            i_ = output_generators[k][i]
            if i_ not in colors:
                dfs(i_, colors, color_idx, output_generators)

    assert len(output_generators) > 0
    color_idx = 0
    colors = {}
    for i in range(len(output_generators[0])):
        if i not in colors:
            dfs(i, colors, color_idx, output_generators)
            color_idx += 1
    return colors


class EquivariantVector(nn.Module):
    def __init__(self, out_generators, out_channels=1) -> None:
        super().__init__()
        self.out_generators = out_generators
        self.out_features = len(out_generators[0])
        self.out_generators = deepcopy(out_generators)
        self.out_channels = out_channels
        self.colors_v = create_colored_vector(out_generators)
        self.num_colors_v = len(set(self.colors_v.values()))
        self.num_weights = self.num_colors_v * out_channels

        idx_vector = np.zeros((self.out_features * out_channels,), dtype=int)
        for i in range(out_channels):
            row_base = i * self.out_features
            v_base = i * self.num_colors_v
            for k, v in self.colors_v.items():
                idx_vector[row_base + k] = v_base + v
        self.register_buffer("idx_vector", torch.tensor(idx_vector, dtype=torch.long))

    def forward(self, X):
        return X[self.idx_vector]

    def right_inverse(self, x):
        # Calculate sums and counts for each unique bias
        sums = torch.bincount(self.idx_vector, weights=x, minlength=self.num_weights)
        counts = torch.bincount(self.idx_vector, minlength=self.num_weights)

        # Avoid division by zero
        counts = counts.clamp(min=1)

        # Compute the average value for each unique bias
        unique_vector = sums / counts.to(sums.dtype)
        return unique_vector


class EquivariantMatrix(nn.Module):
    def __init__(
        self, in_generators, out_generators=None, in_channels=1, out_channels=1
    ):
        super().__init__()
        self.out_generators = out_generators
        self.in_generators = in_generators
        self.in_features = len(in_generators[0])
        self.out_features = len(out_generators[0])
        self.in_generators = deepcopy(in_generators)
        self.out_generators = deepcopy(out_generators)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.colors_mat = create_colored_matrix(in_generators, out_generators)
        self.num_colors_mat = len(set(self.colors_mat.values()))
        self.num_weights = self.num_colors_mat * in_channels * out_channels

        idx_matrix = np.zeros(
            (self.out_features * out_channels, self.in_features * in_channels),
            dtype=int,
        )
        for i in range(out_channels):
            row_base = i * self.out_features
            for j in range(in_channels):
                col_base = j * self.in_features
                v_base = (i * in_channels + j) * self.num_colors_mat
                for k, v in self.colors_mat.items():
                    idx_matrix[row_base + k[1], col_base + k[0]] = v_base + v
        self.register_buffer("idx_matrix", torch.tensor(idx_matrix, dtype=torch.long))

        # basis_cat = torch.stack(
        #     [
        #         torch.from_numpy((idx_matrix == i).flatten()).to(torch.float)
        #         for i in range(self.num_weights + 1)
        #     ],
        #     dim=-2,
        # )
        # # Il faut la concatener de la bonne manière pour ensuite en faire l'inverse
        # torch.linalg.pinv(basis_cat)

    def forward(self, X):
        return X[self.idx_matrix]

    # def right_inverse(self, A):
    #     # We first vectorize A
    #     A_vec = A.resize(
    #         self.out_features * self.out_channels, self.in_features * self.in_channels
    #     )
    #     return torch.matmul(A_vec, self.inv_parametrization)

    def right_inverse(self, x):
        # Flatten the index and value tensors
        flat_idx = self.idx_matrix.flatten()
        flat_x = x.flatten()

        # Calculate sums and counts for each unique weight
        sums = torch.bincount(flat_idx, weights=flat_x, minlength=self.num_weights)
        counts = torch.bincount(flat_idx, minlength=self.num_weights)

        # Avoid division by zero by clamping counts to a minimum of 1
        # The sum for an un-indexed parameter will be 0, so the result is correct
        counts = counts.clamp(min=1)

        # Compute the average value for each unique weight
        unique_weights = sums / counts.to(sums.dtype)
        return unique_weights


if __name__ == "__main__":

    generators = equivariant_permutations_inputs(["a", "a", "b", "b", "a"])
    bias = EquivariantVector(generators)
    print(bias.idx_vector)
