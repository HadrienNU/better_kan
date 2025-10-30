import torch
import copy

from .layers import build_layer


def build_KAN(
    layers_types,
    layers_hidden,
    add_batch_norm=False,
    **kwargs,
):
    """
    Helper function to build stack of KAN layers
    """
    if not isinstance(layers_types, list):
        layers_types = [layers_types] * (len(layers_hidden) - 1)
    layers = torch.nn.ModuleList()
    for layer_cls, in_features, out_features in zip(layers_types, layers_hidden, layers_hidden[1:]):
        if add_batch_norm:
            layers.append(torch.nn.BatchNorm1d(in_features))  # Normalize input
        layer = build_layer(in_features, out_features, layer_cls, **kwargs)
        layers.append(layer)
    return KAN(layers)


class KAN(torch.nn.Module):
    def __init__(self, layers):
        super(KAN, self).__init__()
        self.layers = layers

    @property
    def width(self):
        return [self.layers[0].in_features] + [la.out_features for la in self.layers]

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def update_grid(self, x, grid_size=-1):
        for layer in self.layers:
            if hasattr(layer, "update_grid"):
                layer.update_grid(x, grid_size=grid_size)
            if x is not None:
                x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers if hasattr(layer, "regularization_loss"))

    def prune(self, threshold=1e-2, mode="auto", active_neurons_id=None):
        """
        pruning KAN on the node level. If a node has small incoming or outgoing connection, it will be pruned away.

        Current KAN is unchanged but it return a new pruned model.

        Args:
        -----
            threshold : float
                the threshold used to determine whether a node is small enough
            mode : str
                "auto", "highest" or "manual". If "auto", the thresold will be used to automatically prune away nodes. If "highest", active If "manual", active_neuron_id give the number of neurons to keep at each hidden layer is needed to specify which neurons are kept (others are thrown away).
            active_neuron_id : list of id lists or list of number of neurons to keep
                For example, in "manual" mode [[0,1],[0,2,3]] means keeping the 0/1 neuron in the 1st hidden layer and the 0/2/3 neuron in the 2nd hidden layer. Pruning input and output neurons is not supported yet.
                In "highest" mode [2,3] means keep 2 neurons in the first hidden layer and 3 in the second ones

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
                elif mode == "highest":
                    inout_important = torch.max(self.layers[i].l1_norm, dim=1)[0] + torch.max(self.layers[i + 1].l1_norm, dim=0)[0]
                    overall_important = torch.zeros(self.width[i + 1], dtype=torch.bool)
                    overall_important[torch.argsort(inout_important, descending=True)[: active_neurons_id[i]]] = True
                elif mode == "manual":
                    overall_important = torch.zeros(self.width[i + 1], dtype=torch.bool)
                    overall_important[active_neurons_id[i]] = True
                active_neurons.append(torch.where(overall_important == True)[0].to(device=self.layers[i].bias.device))
            else:
                active_neurons.append(list(range(self.width[i + 1])))
        active_neurons.append(list(range(self.width[-1])))  # Output size

        new_layers = torch.nn.ModuleList()
        for i in range(len(self.width) - 1):
            # new_layer = copy.deepcopy(self.layers[i])
            if hasattr(self.layers[i], "get_inout_subset"):
                self.layers[i].get_inout_subset(active_neurons[i], active_neurons[i + 1])
            # new_layers.append(new_layer)

        return self
