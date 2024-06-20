import torch
import numpy as np
from .layers import RBFKANLayer, SplinesKANLayer


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
    return mask


# Two helpers functions to build stack  of KAN layers


def build_rbf_layers(
    layers_hidden,
    permutation_invariants=None,
    **kwargs,
):
    masks = [None] * (len(layers_hidden) - 1)
    if permutation_invariants is not None:
        masks[0] = invariant_input(permutation_invariants)
    layers = torch.nn.ModuleList()
    for input_dim, output_dim, mask in zip(layers_hidden, layers_hidden[1:], masks):
        layers.append(
            RBFKANLayer(
                input_dim,
                output_dim,
                mask=mask,
                **kwargs,
            )
        )
    return KAN(layers)


def build_splines_layers(
    layers_hidden,
    permutation_invariants=None,
    **kwargs,
):
    masks = [None] * (len(layers_hidden) - 1)
    if permutation_invariants is not None:
        masks[0] = invariant_input(permutation_invariants)
    layers = torch.nn.ModuleList()
    for input_dim, output_dim, mask in zip(layers_hidden, layers_hidden[1:], masks):
        layers.append(
            SplinesKANLayer(
                input_dim,
                output_dim,
                mask=mask,
                **kwargs,
            )
        )
    return KAN(layers)


class KAN(torch.nn.Module):
    def __init__(self, layers):
        super(KAN, self).__init__()

        self.depth = len(layers)
        self.width = [layers[0].input_dim] + [la.output_dim for la in layers]
        self.layers = layers

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)
