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


def equivariant_inputs(inputs, is_matrix=False):
    """
    Return the symmetry group for a vectorial or a matricial input
    inputs is an array that contain the type of the entry, i.e. equivariant entry share the same type
    
    Return a sympy PermutationGroup
    """

    inputs = np.asarray(inputs).ravel()
    nb_inputs = len(inputs)
    types = np.unique(inputs)
    if len(types) == nb_inputs:
        return trivial_group(nb_inputs)
    groups = []
    for t in types:
        if np.sum(inputs==g) > 1: # Only if more than one entry share the same type
            inds= np.nonzero(inputs==g)[0]
            cycle= Permutation([inds], size=nb_inputs)
            perm=Permutation([inds[:2]], size=nb_inputs)
            groups.append(PermutationGroup(cycle,perm))
    return DirectProduct(*groups)


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
    def __init__(self, out_group, out_channels=1) -> None:
        super().__init__()
        self.out_group = out_group
        out_generators = [perm.array_form for perm in out_group.generators]
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
        self.register_buffer('idx_vector', torch.tensor(idx_vector, dtype=torch.long))
        basis_cat = torch.stack([(idx_vector==i).to(torch.float) for i in range(self.num_weights+1)], dim=-2)
        # Il faut la concatener de la bonne manière pour ensuite en faire l'inverse
        self.register_buffer("inv_parametrization", torch.linalg.pinv(basis_cat))

    def forward(self, X):
        return X[self.idx_vector]

    def right_inverse(self, A):
        # Doit faire la moyenne des poids en commun, ou juste prendre un éléments
        return torch.matmul(A, self.inv_parametrization)


class EquivariantMatrix(nn.Module):
    def __init__(self, in_group, in_channels= 1, out_group=None, out_channels=1):
        super().__init__()
        if out_group is None:
            out_group = trivial_group(out_channels)
        self.out_group = out_group
        self.in_group = in_group
        out_generators = [perm.array_form for perm in out_group.generators]
        in_generators = [perm.array_form for perm in in_group.generators]
        self.in_features = len(in_generators[0])
        self.out_features = len(out_generators[0])
        self.in_generators = deepcopy(in_generators)
        self.out_generators = deepcopy(out_generators)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.colors_mat = create_colored_matrix(in_generators, out_generators)
        self.num_colors_mat = len(set(self.colors_mat.values()))
        self.num_weights = self.num_colors_mat * in_channels * out_channels
        self.matrix = nn.Parameter(torch.Tensor(self.num_weights_mat))

        idx_matrix = np.zeros((self.out_features * out_channels, self.in_features * in_channels), dtype=int)
        for i in range(out_channels):
            row_base = i * self.out_features
            for j in range(in_channels):
                col_base = j * self.in_features
                v_base = (i * in_channels + j) * self.num_colors_mat
                for k, v in self.colors_mat.items():
                    idx_matrix[row_base + k[1], col_base + k[0]] = v_base + v
        self.register_buffer('idx_matrix', torch.tensor(idx_matrix, dtype=torch.long))
        
        basis_cat = torch.stack([(idx_matrix==i).to(torch.float).flatten() for i in range(self.num_weights+1)], dim=-2)
        # Il faut la concatener de la bonne manière pour ensuite en faire l'inverse
        self.register_buffer("inv_parametrization", torch.linalg.pinv(basis_cat))

    def forward(self, X):
        return X[self.idx_matrix]

    def right_inverse(self, A):
        # We first vectorize A
        A_vec= A.resize(self.out_features * self.out_channels, self.in_features * self.in_channels)
        return torch.matmul(A_vec, self.inv_parametrization)


def invariant_input(inputs):
    """
    Build the mask matrix for invariant input

    inputs is an array that contain 0 if entry is independent, -1 if input is discarded and the group number if entry has to be grouped with another
    """
    inputs = np.asarray(inputs).ravel()
    groups = np.unique(inputs[inputs > 0])
    nb_indep_entry = len(groups) + len(inputs[inputs <= 0])
    mask = torch.zeros(nb_indep_entry, len(inputs))
    m = len(groups)
    for n, i in enumerate(inputs):
        if i == 0:
            mask[m, n] = 1
            m += 1
        elif i > 0:
            mask[i - 1, n] = 1
        else:
            m += 1
    return mask[mask.sum(1).to(dtype=torch.bool), :]  # Remove non contributing features


def mask_subset(layer, in_id=None):
    """
    Get new mask for layer subset
    """
    if parametrize.is_parametrized(layer, "grid"):
        mask = layer.parametrizations["grid"][0].mask
        if in_id is not None:
            mask = mask[:, in_id]
        return mask[mask.sum(1).to(dtype=torch.bool), :]  # Remove entry that does not matter
    else:
        return None


class InputMask(nn.Module):
    def __init__(self, mask) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.register_buffer("inv_mask", torch.linalg.pinv(mask))
        self.reduced_in_dim = mask.shape[0]

    def forward(self, X):
        return torch.matmul(X, self.mask)

    def right_inverse(self, A):
        return torch.matmul(A, self.inv_mask)


class GridMask(InputMask):  # Right inverse method is a bit different
    def __init__(self, mask) -> None:
        super().__init__(mask)

    def right_inverse(self, A):
        # A shape, (grid_size, in_features)
        new_grid = torch.empty(A.shape[0], self.reduced_in_dim, device=A.device)  # shape (grid_size, reduced_in_dim)
        # Group together values from the same group
        for n in range(self.reduced_in_dim):
            x_sorted = torch.sort(torch.flatten(A[:, self.mask.to(dtype=torch.bool)[n, :]]), dim=0)[0]
            new_grid[:, n] = x_sorted[torch.linspace(0, x_sorted.shape[0] - 1, A.shape[0], dtype=torch.int64, device=A.device)]
        return new_grid
