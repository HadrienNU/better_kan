"""
Set of modules and function to deal with permutation invariance of the input
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from itertools import product
from collections.abc import Sequence

from sympy.combinatorics import Permutation, PermutationGroup


def parametrize_kan_equivariance(model, equiv_list):
    """
    Set the prametrization for the entire network
    """


def parametrize_layer_equivariance(layer, equiv):
    """
    Take a KANLayer and parametrize what is needed.
    """


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


def draw_matrix_parametrizations(parametrizations, cmap="rainbow", markersize=20):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmx
    import pylab as mpl

    wvals = np.arange(parametrizations.num_weights)
    w = parametrizations.forward(wvals)
    if len(w.shape) == 1:
        w = w.reshape(-1, 1)
    mrklist = ["3", "d", ",", "2", "8", "s", "|", "^", "<", "4", "H", "+", "o", "v", "p", "D", ">", "1", ".", "_", "*", "h"]
    cm = plt.get_cmap("Greys")
    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=len(wvals) - 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    def plot_single_matrix(ax, matrix):
        if markersize > 0:
            for wval in wvals:
                x, y = np.nonzero(matrix == wval)
                x = w.shape[0] - x - 1
                plt.scatter(x, y, s=markersize, marker=mrklist[wval % len(mrklist)], color=scalarMap.to_rgba(wval), alpha=1)
        ax.imshow(matrix, cmap=cmap, alpha=0.8, vmin=0, vmax=len(wvals) - 1, origin="lower", aspect="equal")
        ax.set_xticks(np.arange(matrix.shape[1] - 1) + 0.5, [])
        ax.set_yticks(np.arange(matrix.shape[0] - 1) + 0.5, [])
        ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
        ax.set_ylim(-0.5, matrix.shape[0] - 0.5)
        ax.grid(which="major", color="grey", linestyle="-", alpha=0.5)
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")

    # If w is a high-dimensional tensor, treat it as a collection of matrices
    if len(w.shape) > 2:
        num_matrices = w.shape[0]
        # Auto-calculate grid size for subplots
        ncols = num_matrices
        nrows = 1

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        # Flatten axes array for easy iteration
        axes = axes.flatten()

        for i in range(num_matrices):
            sub_matrix = w[i]
            ax = axes[i]
            plot_single_matrix(ax, sub_matrix)
            ax.set_title(f"Submatrix {i}")

        # Hide any unused subplots
        for i in range(num_matrices, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

    else:  # Original behavior for 2D or 1D arrays
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_single_matrix(ax, w)


def create_colored_tensor(shape: tuple[int, ...], actions: list[list[torch.Tensor] | None]) -> tuple[torch.Tensor, int]:
    """
    Determines the unique parameters of a tensor under group actions on its axes.

    This version correctly handles independent group actions on different axes, where
    each action may be defined by a different number of generators. It also robustly
    handles generators passed as Python lists.

    Args:
        shape (tuple[int, ...]): The shape of the tensor.
        actions (list[list[torch.Tensor] | None]): A list of actions for each axis.

    Returns:
        tuple[torch.Tensor, int]:
            - A LongTensor of the given `shape` with integer "colors".
            - The total number of unique colors (parameters).
    """
    if len(shape) != len(actions):
        raise ValueError("The length of shape and actions must be the same.")

    # 1. Separate dimensions into active (have actions) and stacked (None action)
    active_dims, stack_dims = [], []
    active_shape, stack_shape = [], []
    active_actions = []

    for i, (size, action) in enumerate(zip(shape, actions)):
        if action is None:
            stack_dims.append(i)
            stack_shape.append(size)
        else:
            active_dims.append(i)
            active_shape.append(size)
            # *** FIX: Ensure all generators are tensors for consistent indexing ***
            processed_action = [torch.as_tensor(gen, dtype=torch.long) for gen in action]
            active_actions.append(processed_action)

    # 2. Color the smaller, active subspace first
    if not active_dims:
        num_colors = np.prod(shape).item()
        return torch.arange(num_colors).view(shape), num_colors

    active_shape = tuple(active_shape)
    sub_colors = torch.full(active_shape, -1, dtype=torch.long)
    num_sub_colors = 0

    for initial_pos in product(*(range(s) for s in active_shape)):
        if sub_colors[initial_pos] == -1:
            q = [initial_pos]
            sub_colors[initial_pos] = num_sub_colors
            head = 0
            while head < len(q):
                current_pos = q[head]
                head += 1

                for dim_idx, generators_for_dim in enumerate(active_actions):
                    for gen in generators_for_dim:
                        next_pos_list = list(current_pos)
                        original_coord = current_pos[dim_idx]
                        # Now gen is guaranteed to be a tensor, so .item() is safe
                        next_pos_list[dim_idx] = gen[original_coord].item()

                        next_pos = tuple(next_pos_list)
                        if sub_colors[next_pos] == -1:
                            sub_colors[next_pos] = num_sub_colors
                            q.append(next_pos)
            num_sub_colors += 1

    # 3. Expand the coloring across the stacked dimensions
    total_colors = num_sub_colors * np.prod(stack_shape).item()
    final_colors_tensor = torch.empty(shape, dtype=torch.long)

    offset = 0
    for stack_pos in product(*(range(s) for s in stack_shape)):
        slice_indices = [slice(None)] * len(shape)
        for i, idx in zip(stack_dims, stack_pos):
            slice_indices[i] = idx

        final_colors_tensor[tuple(slice_indices)] = sub_colors + offset
        offset += num_sub_colors

    return final_colors_tensor, total_colors


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

        idx_weight = np.zeros((self.out_features * out_channels,), dtype=int)
        for i in range(out_channels):
            row_base = i * self.out_features
            v_base = i * self.num_colors_v
            for k, v in self.colors_v.items():
                idx_weight[row_base + k] = v_base + v
        self.register_buffer("idx_weight", torch.tensor(idx_weight, dtype=torch.long))

    def forward(self, X):
        return X[self.idx_weight]

    def right_inverse(self, x):
        # Calculate sums and counts for each unique bias
        sums = torch.bincount(self.idx_weight, weights=x, minlength=self.num_weights)
        counts = torch.bincount(self.idx_weight, minlength=self.num_weights)

        # Avoid division by zero
        counts = counts.clamp(min=1)

        # Compute the average value for each unique bias
        unique_vector = sums / counts.to(sums.dtype)
        return unique_vector


class EquivariantMatrix(nn.Module):
    def __init__(self, in_generators, out_generators=None, in_channels=1, out_channels=1):
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

        idx_weight = np.zeros(
            (self.out_features * out_channels, self.in_features * in_channels),
            dtype=int,
        )
        for i in range(out_channels):
            row_base = i * self.out_features
            for j in range(in_channels):
                col_base = j * self.in_features
                v_base = (i * in_channels + j) * self.num_colors_mat
                for k, v in self.colors_mat.items():
                    idx_weight[row_base + k[1], col_base + k[0]] = v_base + v
        self.register_buffer("idx_weight", torch.tensor(idx_weight, dtype=torch.long))

    def forward(self, X):
        return X[self.idx_weight]

    def right_inverse(self, x):
        # Flatten the index and value tensors
        flat_idx = self.idx_weight.flatten()
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


class EquivariantParametrization(nn.Module):
    """
    A generic parametrization for creating any equivariant tensor.

    This module constrains a tensor to be equivariant under specified group
    actions on its axes. It does this by identifying the minimal set of unique
    parameters and mapping them to the full tensor.

    Args:
        shape (tuple[int, ...]): The shape of the tensor.
        actions (list): A list of group actions (generators) for each axis.
                        Use `None` for axes not affected by the group.
    """

    def __init__(self, shape: tuple[int, ...], actions: list):
        super().__init__()

        self.shape = shape

        # Check that generators length match tensor shape
        for i, s in enumerate(self.shape):
            if actions[i] is not None:
                assert len(actions[i][0]) == s

        idx_tensor, num_colors = create_colored_tensor(shape, actions)
        self.num_weights = num_colors
        self.register_buffer("idx_tensor", idx_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if x.numel() != self.num_weights:
        #     raise ValueError(f"Expected {self.num_unique_params} unique parameters, but got {x.numel()}")
        full_tensor = x[self.idx_tensor]
        print(self.shape, full_tensor.shape)
        return full_tensor  # .view(self.shape)

    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten both index and value tensors for bincount
        flat_idx = self.idx_tensor.flatten()
        flat_x = x.flatten()

        # Calculate sums and counts for each unique parameter
        sums = torch.bincount(flat_idx, weights=flat_x, minlength=self.weight.numel())
        counts = torch.bincount(flat_idx, minlength=self.weight.numel())

        counts = counts.clamp(min=1)  # Avoid division by zero
        return sums / counts.to(sums.dtype)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    generators = equivariant_permutations_inputs(["a", "a", "b", "b", "a"])

    parametrization_tensor = EquivariantMatrix(generators, generators)
    print(parametrization_tensor.idx_weight, parametrization_tensor.num_weights)
    draw_matrix_parametrizations(parametrization_tensor)
    plt.show()
    # parametrization_tensor = EquivariantParametrization((5, 1, 5), [generators, None, None])
    # print(parametrization_tensor.idx_tensor)

    # draw_matrix_parametrizations(parametrization_tensor)
    # plt.show()
