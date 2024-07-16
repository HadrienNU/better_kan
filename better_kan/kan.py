import torch
import copy
from .layers import RBFKANLayer, SplinesKANLayer
from .polynomial_layers import ChebyshevKANLayer
from .permutations import invariant_input

# Two helpers functions to build stack  of KAN layers


def build_rbf_layers(
    layers_hidden,
    permutation_invariants=None,
    add_batch_norm=False,
    **kwargs,
):
    return build_layers([RBFKANLayer] * (len(layers_hidden) - 1), layers_hidden, permutation_invariants, add_batch_norm, **kwargs)


def build_splines_layers(
    layers_hidden,
    permutation_invariants=None,
    add_batch_norm=False,
    **kwargs,
):
    return build_layers([SplinesKANLayer] * (len(layers_hidden) - 1), layers_hidden, permutation_invariants, add_batch_norm, **kwargs)


def build_chebyshev_layers(
    layers_hidden,
    permutation_invariants=None,
    add_batch_norm=False,
    **kwargs,
):
    return build_layers([ChebyshevKANLayer] * (len(layers_hidden) - 1), layers_hidden, permutation_invariants, add_batch_norm, **kwargs)


def build_layers(
    layers_types,
    layers_hidden,
    permutation_invariants=None,
    add_batch_norm=False,
    **kwargs,
):
    masks = [None] * (len(layers_hidden) - 1)
    if permutation_invariants is not None:
        masks[0] = invariant_input(permutation_invariants)
    layers = torch.nn.ModuleList()
    for layer_cls, in_features, out_features, mask in zip(layers_types, layers_hidden, layers_hidden[1:], masks):
        if add_batch_norm:
            layers.append(torch.nn.BatchNorm1d(in_features))  # Normalize input
        layers.append(
            layer_cls(
                in_features,
                out_features,
                mask=mask,
                **kwargs,
            )
        )
    return layers


class KAN(torch.nn.Module):
    def __init__(self, layers):
        super(KAN, self).__init__()
        self.layers = layers

    @property
    def width(self):
        return [self.layers[0].in_features] + [la.out_features for la in self.layers]

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                if hasattr(layer, "update_grid"):
                    layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers if hasattr(layer, "regularization_loss"))

    def initialize_from_another_model(self, other):
        """
        Initialize the grid and weight of current model from another one
        """
        for n, layer in enumerate(self.layers):
            if hasattr(layer, "set_from_another_layer"):
                layer.set_from_another_layer(other.layers[n])
            else:
                layer = copy.deepcopy(other.layers[n])
        return self

    def prune(self, threshold=1e-2, mode="auto", active_neurons_id=None):
        """
        pruning KAN on the node level. If a node has small incoming or outgoing connection, it will be pruned away.

        Current KAN is unchanged but it return a new pruned model.

        Args:
        -----
            threshold : float
                the threshold used to determine whether a node is small enough
            mode : str
                "auto" or "manual". If "auto", the thresold will be used to automatically prune away nodes. If "manual", active_neuron_id is needed to specify which neurons are kept (others are thrown away).
            active_neuron_id : list of id lists
                For example, [[0,1],[0,2,3]] means keeping the 0/1 neuron in the 1st hidden layer and the 0/2/3 neuron in the 2nd hidden layer. Pruning input and output neurons is not supported yet.

        Returns:
        --------
            model2 : KAN
                pruned model

        Example
        -------
        >>> # for more interactive examples, please see demos
        >>> pruned_mode=model.prune()
        >>> pruned_model(dataset["test_input"])
        >>> plot(pruned_mode)
        """

        active_neurons = [list(range(self.width[0]))]  # Input size
        for i in range(len(self.layers) - 1):  # Not considering first and last layers
            if hasattr(self.layers[i], "l1_norm") and hasattr(self.layers[i + 1], "l1_norm"):  # Skip layer that are not KAN
                if mode == "auto":
                    in_important = torch.max(self.layers[i].l1_norm, dim=1)[0] > threshold
                    out_important = torch.max(self.layers[i + 1].l1_norm, dim=0)[0] > threshold
                    overall_important = in_important * out_important
                elif mode == "manual":
                    overall_important = torch.zeros(self.width[i + 1], dtype=torch.bool)
                    overall_important[active_neurons_id[i]] = True
                active_neurons.append(torch.where(overall_important == True)[0].to(device=self.layers[i].weights.device))
            else:
                active_neurons.append(list(range(self.width[i + 1])))
        active_neurons.append(list(range(self.width[-1])))  # Output size

        new_layers = torch.nn.ModuleList()
        for i in range(len(self.width) - 1):
            if hasattr(self.layers[i], "get_subset"):
                new_layers.append(self.layers[i].get_subset(active_neurons[i], active_neurons[i + 1]))
            else:
                new_layers.append(copy.deepcopy(self.layers[i]))

        return KAN(new_layers)
