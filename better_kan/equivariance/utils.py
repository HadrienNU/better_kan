import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import hashlib


def consistent_hash(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest(), 16)


def draw_matrix_parametrizations(parametrizations, X, cmap="rainbow", markersize=20):

    weights = parametrizations.right_inverse(X)

    num_weights = weights.numel()
    weights_shape = weights.shape

    wvals = torch.arange(num_weights, dtype=torch.float).reshape(weights_shape)
    w = parametrizations.forward(wvals).numpy()
    if len(w.shape) == 1:
        w = w.reshape(-1, 1)

    def plot_single_matrix(ax, matrix):
        ax.imshow(matrix, cmap=cmap, alpha=0.8, vmin=w.min(), vmax=w.max(), origin="lower", aspect="equal")
        ax.set_xticks(np.arange(matrix.shape[1] - 1) + 0.5, [])
        ax.set_yticks(np.arange(matrix.shape[0] - 1) + 0.5, [])
        ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
        ax.set_ylim(-0.5, matrix.shape[0] - 0.5)
        ax.grid(which="major", color="grey", linestyle="-")
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
