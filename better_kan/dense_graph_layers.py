import torch
import torch.nn as nn
from torch_scatter import scatter_add

bell_numbers = [1, 1, 2, 5, 15, 52, 203]
bell_numbers_without_self_loops = [1, 1, 2, 3, 8]


class DenseGraphConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_linears,
        n_bias,
        bias=None,
    ):
        super(DenseGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linears = nn.ModuleList([nn.Linear(self.in_channels, self.out_channels, bias=False) for _ in range(n_linears)])
        if bias is not None:
            self.bias = torch.nn.Parameter(bias * torch.ones(out_channels, n_bias))
        else:
            self.register_buffer("bias", torch.zeros(out_channels, n_bias))

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()
        if self.bias.requires_grad:  # If bias is trainable
            nn.init.uniform_(self.bias)


class DenseGraphConvNodeToNode(DenseGraphConv):
    def __init__(self, in_channels, out_channels, bias=None):
        super().__init__(in_channels, out_channels, 2, 1, bias)

    def equivariant_basis(self, X, mask=None):
        if mask is None:
            agg = X.sum(dim=1, keepdim=True)
        else:  # Sum but taking mask into account
            agg = (X * mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
        return X, agg.expand_as(X)

    def forward(self, x, adj, mask=None, edge_attrs=None):

        out = self.equivariant_basis(x, mask)
        bias = self.bias[:, 0]
        if mask is not None:
            bias = bias * mask.unsqueeze(-1)
        return sum(self.linears[i](x) for i, x in enumerate(out)) + bias


class DenseGraphConvEdgeToNode(DenseGraphConv):
    def __init__(self, in_channels, out_channels, bias=None, include_self_loops=False):
        if include_self_loops:
            n_lin = 5
            n_bias = 1
        else:
            n_lin = 3
            n_bias = 1
        super().__init__(in_channels, out_channels, n_lin, n_bias, bias)
        self.include_self_loops = include_self_loops

    def equivariant_basis(self, E, mask):  # Taille 5

        if mask is not None:
            mask_edge = mask.unsqueeze(1) * mask.unsqueeze(2)
            E = E * mask_edge.unsqueeze(-1)

        Esum_1 = E.sum(dim=1)

        Esum_2 = E.sum(dim=2)

        Esum_all = E.sum(1).sum(dim=2, keepdim=True).expand_as(Esum_1)

        if self.include_self_loops:
            E_diag = E.diagonal(dim1=1, dim2=2).transpose(1, 2)
            E_diag_sum = E_diag.sum(dim=1, keepdim=True).expand_as(Esum_1)
            return E_diag, Esum_1, Esum_2, Esum_all, E_diag_sum
        else:
            return Esum_1, Esum_2, Esum_all

    def forward(self, x, adj, mask=None, edge_attrs=None):

        out = self.equivariant_basis(edge_attrs, mask)  # Compute the basis part
        bias = self.bias[:, 0]
        if mask is not None:
            bias = bias * mask.unsqueeze(-1)
        return sum(self.linears[i](x) for i, x in enumerate(out)) + bias


class DenseGraphConvNodeToEdge(DenseGraphConv):
    def __init__(self, in_channels, out_channels, bias=None, include_self_loops=False):
        if include_self_loops:
            n_lin = 5
            n_bias = 2
        else:
            n_lin = 3
            n_bias = 1
        super().__init__(in_channels, out_channels, n_lin, n_bias, bias)
        self.include_self_loops = include_self_loops

    def equivariant_basis(self, X, mask=None):
        # Taille 5
        _, N, _ = X.shape
        X_columns = X.unsqueeze(1).expand(-1, N, -1, -1)
        X_rows = X.unsqueeze(2).expand_as(X_columns)
        sum_all = X.sum(dim=1, keepdim=True)
        X_sum = sum_all.unsqueeze(1).expand_as(X_columns)

        if self.include_self_loops:

            X_diag = torch.diag_embed(X.transpose(1, 2)).permute(0, 2, 3, 1)
            X_diag_sum = torch.diag_embed(sum_all.expand_as(X).transpose(1, 2)).permute(0, 2, 3, 1)
            return X_columns, X_rows, X_sum, X_diag, X_diag_sum
        else:
            return X_columns, X_rows, X_sum

    def forward(self, x, adj, mask=None, edge_attrs=None):

        out = self.equivariant_basis(x, mask)  # Compute the basis part
        bias = self.bias[:, 0]
        if self.include_self_loops:
            bias += (torch.ones_like(x[:, :, 0]) - 2 * torch.diag(torch.ones_like(x[:, 0, 0]))).unsqueeze(-1) * self.bias[:, 1]
        if mask is not None:
            mask_edge = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            bias = bias * mask_edge.unsqueeze(-1)  # Es-ce que ça marche pour le deuxième bias?
        return sum(self.linears[i](x) for i, x in enumerate(out)) + bias  # Et pour l'autre biais il faut y mettre un pattern un peu plus complexe


class DenseGraphConvEdgeToEdge(DenseGraphConv):
    def __init__(self, in_channels, out_channels, bias=None, include_self_loops=False):
        if include_self_loops:
            n_lin = 15
            n_bias = 2
        else:
            n_lin = 7
            n_bias = 1
        super().__init__(in_channels, out_channels, n_lin, n_bias, bias)
        self.include_self_loops = include_self_loops

    def equivariant_basis(self, E, mask=None):
        # Size 7 or 15

        # Sum over columns (dim 2) -> result depends on rows
        sum_rows = E.sum(dim=2, keepdim=True)  # B x N x 1 x D
        # Sum over rows (dim 1) -> result depends on cols
        sum_cols = E.sum(dim=1, keepdim=True)  # B x 1x N x D
        # Global sum
        sum_all = E.sum(dim=(1, 2), keepdim=True)  # B x 1 x 1 D

        rows_to_rows = sum_rows.expand_as(E)
        cols_to_cols = sum_cols.expand_as(E)
        sumall_to_E = sum_all.expand_as(E)

        # Op 9: Tile sum_rows (N, m, D) along dim 1
        rows_to_cols = sum_rows.transpose(1, 2).expand_as(E)
        cols_to_rows = sum_cols.transpose(1, 2).expand_as(E)

        transpose = torch.transpose(E, 1, 2)

        if self.include_self_loops:
            diag_part = torch.diagonal(E, dim1=1, dim2=2).transpose(1, 2)  # B x N x D
            sum_diag_part = diag_part.sum(dim=1, keepdim=True)  # B x 1 x D

            # Helper function to create diagonal matrix B x N x N x D from B x N x D
            def diag_embed_channels_last(t):
                # t: B x N x D
                # permute to B x D x N -> diag_embed -> B x D x N x N
                # permute back to B x N x N x D
                return torch.diag_embed(t.transpose(1, 2)).permute(0, 2, 3, 1)

            diag_to_diag = diag_embed_channels_last(diag_part)

            rows_to_diag = diag_embed_channels_last(sum_rows[:, :, 0, :])
            cols_to_diag = diag_embed_channels_last(sum_cols[:, 0, :, :])

            sum_all_to_diag = diag_embed_channels_last(sum_all[:, 0, :, :].expand_as(diag_part))
            diag_to_rows = diag_part.unsqueeze(2).expand_as(E)
            diag_to_cols = diag_part.unsqueeze(1).expand_as(E)

            diag_sum_to_diag = diag_embed_channels_last(sum_diag_part.expand_as(diag_part))
            diagsum_to_E = sum_diag_part.unsqueeze(1).expand_as(E)

            return (
                E,
                transpose,
                rows_to_rows,
                cols_to_cols,
                rows_to_cols,
                cols_to_rows,
                sumall_to_E,
                diag_to_diag,
                rows_to_diag,
                cols_to_diag,
                sum_all_to_diag,
                diag_to_rows,
                diag_to_cols,
                diag_sum_to_diag,
                diagsum_to_E,
            )

        else:
            return E, transpose, rows_to_rows, cols_to_cols, rows_to_cols, cols_to_rows, sumall_to_E

    def forward(self, x, adj, mask=None, edge_attrs=None):

        out = self.equivariant_basis(edge_attrs, mask)  # Compute the basis part
        bias = self.bias[:, 0]
        if self.include_self_loops:
            bias += (torch.ones_like(x[:, :, 0]) - 2 * torch.diag(torch.ones_like(x[:, 0, 0]))).unsqueeze(-1) * self.bias[:, 1]
        if mask is not None:
            bias = bias * mask.unsqueeze(-1).unsqueeze(-1)  # Es-ce que ça marche pour le deuxième bias?
        return sum(self.linears[i](x) for i, x in enumerate(out)) + bias  # Et pour l'autre biais il faut y mettre un pattern un peu plus complexe
