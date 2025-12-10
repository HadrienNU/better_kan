import torch
import torch.nn as nn
from torch_scatter import scatter_add

bell_numbers = [1, 1, 2, 5, 15, 52, 203]


class GraphConv(nn.Module):
    def __init__(
        self,
        in_type,
        out_type,
        in_channels,
        out_channels,
        bias=None,
    ):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linears = nn.ModuleList([nn.Linear(self.in_channels, self.out_channels, bias=False) for _ in range(bell_numbers[in_type + out_type])])
        if bias is not None:
            self.bias = torch.nn.Parameter(bias * torch.ones(out_channels, bell_numbers[out_type]))
        else:
            self.register_buffer("bias", torch.zeros(out_channels, bell_numbers[out_type]))

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()
        if self.bias.requires_grad:  # If bias is trainable
            nn.init.uniform_(self.bias)

    def forward(self, x, edge_index):

        out = self.equivariant_basis(x, edge_index)  # Compute the basis part
        bias = self.bias[:, 0]
        if self.bias.shape[1] == 2:
            bias += (torch.ones_like(x[:, :, 0]) - 2 * torch.diag(torch.ones_like(x[:, 0, 0]))).unsqueeze(-1) * self.bias[:, 1]

        return sum(self.linears[i](x) for i, x in enumerate(out)) + bias  # Et pour l'autre biais il faut y mettre un pattern un peu plus complexe


class GraphConvNodeNode(GraphConv):
    def __init__(self, in_channels, out_channels, bias=None):
        super().__init__(1, 1, in_channels, out_channels, bias)

    def equivariant_basis(self, X, edge_index):
        # Aggregate with scatter_add
        agg = scatter_add(X[edge_index[0]], edge_index[1], dim=0, dim_size=X.size(0))
        return X, agg


class GraphConvEdgeNode(GraphConv):
    def __init__(self, in_channels, out_channels, bias=None):
        super(GraphConvNodeNode).__init__(1, 1, in_channels, out_channels, bias)

    def equivariant_basis(self, E, edge_index):  # Taille 5

        X_diag = E[edge_index[0] == edge_index[1], :]  # Ce sont tous les élements pour lesquels src == dist

        X_sum = scatter_add(E, edge_index[1], dim=0, dim_size=int(edge_index.max().item() + 1))  # Sum over all edge leading to i

        scatter_add(E, edge_index[0], dim=0, dim_size=X.size(0))

        scatter_add(X[:, edge_index[0]], edge_index[1], dim=0, dim_size=X.size(0))

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)  # Compute the sum over connected value
        return x, x_summed


class GraphConvNodeEdge(GraphConv):
    def __init__(self, in_channels, out_channels, bias=None):
        super(GraphConvNodeNode).__init__(1, 2, in_channels, out_channels, bias)

    def equivariant_basis(self, x, edge_index):
        # Taille 5
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)  # Compute the sum over connected value
        return x, x_summed


class GraphConvEdgeEdge(GraphConv):
    def __init__(self, in_channels, out_channels, bias=None):
        super(GraphConvNodeNode).__init__(1, 2, in_channels, out_channels, bias)

    def equivariant_basis(self, x, edge_index):
        # Taille 15
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)  # Compute the sum over connected value
        return x, x_summed
