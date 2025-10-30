import torch
import torch.nn as nn

from .functions import Grid


# Helpers function to build a layer with only one type of function, for more complex setup write something by yourself
def build_layer(in_features, out_features, functions_type, grid_size=None, bias=None, fast_version=False, pooling_op="sum", pooling_args=None, **fct_kwargs):

    if grid_size is not None:
        grid = Grid(in_features, grid_size)
        fct = functions_type(in_features, out_features, grid, **fct_kwargs)

    else:
        fct = functions_type(in_features, out_features, **fct_kwargs)
    return KANLayer(in_features, out_features, nn.ModuleList([fct]), bias=bias, fast_version=fast_version, pooling_op=pooling_op, pooling_args=pooling_args)


class KANLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        functions,
        bias=None,
        fast_version=False,
        pooling_op="sum",
        pooling_args=None,
    ):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias is not None:
            self.bias = torch.nn.Parameter(bias * torch.ones(out_features))
        else:
            self.bias = torch.zeros(out_features)

        # Run some check on functions, assert that in_features and out_features are all coherent

        self.functions = functions

        if pooling_op.lower() in ["sum", "prod", "logsumexp", "min", "max"]:
            self.pooling_op = pooling_op.lower()
        elif pooling_op.lower() in ["power", "pow"]:
            self.pooling_op = "power"
            assert pooling_args > 0 or pooling_args < 0  # Should raise error if not numeric
            self.pooling_power = pooling_args
        elif pooling_op.lower() == "fsum":
            self.pooling_op = "fsum"
            assert callable(pooling_args["f"]) and callable(pooling_args["invf"])  # The two argument exits and are callabel
            self.pooling_args = pooling_args
        else:
            raise ValueError("Unknown pooling operation")

        self.set_speed_mode(fast_version)

    def forward(self, x: torch.Tensor):
        # Depending if the functions are in slow or fast mode
        if self.fast_mode:
            return torch.stack([fct(x) for fct in self.functions], dim=0).sum(dim=0)
        else:
            original_shape = x.shape
            self.min_vals = torch.min(x, dim=0).values
            self.max_vals = torch.max(x, dim=0).values
            # Somes statistics for regularisation and plot
            out_acts = self.activations_eval(x)
            self.l1_norm = torch.mean(torch.abs(out_acts), dim=0) / (self.max_vals - self.min_vals)  # out_dim x in_dim
            if self.pooling_op == "sum":
                output = torch.sum(out_acts, dim=2)
            elif self.pooling_op == "prod":
                output = torch.prod(out_acts, dim=2)
            elif self.pooling_op == "power":
                output = torch.pow(
                    torch.sum(torch.pow(out_acts, self.pooling_power), dim=2),
                    1 / self.pooling_power,
                )
            elif self.pooling_op == "logsumexp":
                output = torch.logsumexp(out_acts, dim=2)
            elif self.pooling_op == "min":
                output = torch.min(out_acts, dim=2)
            elif self.pooling_op == "max":
                output = torch.max(out_acts, dim=2)
            elif self.pooling_op == "fsum":
                output = self.pooling_args["invf"](torch.sum(self.pooling_args["f"](out_acts), dim=2))
            return (output + self.bias.unsqueeze(0)).view(*original_shape[:-1], self.out_features)

    def activations_eval(self, x):
        """
        Get values of the activations
        """
        return torch.stack([fct.activations_eval(x) for fct in self.functions], dim=0).sum(dim=0)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.
        """
        if hasattr(self, "l1_norm"):
            regularization_loss_activation = self.l1_norm.sum()
            p = self.l1_norm / regularization_loss_activation
            regularization_loss_entropy = -torch.sum(p * torch.log(p + 1e-6))  # Regularization to avoid 0 value
            return regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy
        else:
            return torch.tensor(0.0)

    def update_grid(self, x, grid_size=-1):
        for fct in self.functions:
            if hasattr(fct, "update_grid"):
                fct.update_grid(x, grid_size=grid_size)
        return self

    def get_inout_subset(self, in_id=None, out_id=None):
        """
        set a smaller KANLayer from a larger KANLayer (used for pruning)

        Args:
        -----
            parent : kan_layer
                An input KANLayer to be set as a subset of this one
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons

        Returns:
        --------
            newlayer : KANLayer

        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_small.set_from_another_layer(kanlayer_large,[0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        """
        if in_id is None:
            in_id = torch.arange(self.in_features)
        if out_id is None:
            out_id = torch.arange(self.out_features)
        self.in_features = len(in_id)
        self.out_features = len(out_id)
        self.bias = self.bias[out_id]
        for fct in self.functions:
            fct.get_inout_subset(in_id, out_id)

        return self

    def set_speed_mode(self, fast=True):  # TODO: set reduction to True in functions
        if fast:
            self.fast_mode = True
            if hasattr(self, "l1_norm"):
                del self.l1_norm
            if self.pooling_op != "sum":
                raise ValueError(f"Fast mode is incompatible with pooling operation {self.pooling_op}")
        else:
            self.fast_mode = False
