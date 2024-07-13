"""
Set of modules and function to deal with permutation invariance of the input
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.parametrize as parametrize


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
