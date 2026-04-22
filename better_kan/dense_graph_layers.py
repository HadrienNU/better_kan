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
            agg = X.mean(dim=1, keepdim=True)
        else:  # Sum but taking mask into account
            agg = (X * mask.unsqueeze(-1)).mean(dim=1, keepdim=True)
        return X, agg.expand_as(X)

    # def forward(self, x, adj, mask=None, edge_attrs=None):

    #     out = self.equivariant_basis(x, mask)
    #     bias = self.bias[:, 0]
    #     if mask is not None:
    #         bias = bias * mask.unsqueeze(-1)
    #     return sum(self.linears[i](x) for i, x in enumerate(out)) + bias

    def forward(self, x, adj, mask=None, edge_attrs=None):
        # x shape: (B, N, Din)

        # 1. Project the node features themselves
        out = self.linears[0](x)  # (B, N, Dout)

        # 2. Compute Global Aggregation
        if mask is not None:
            # mask shape: (B, N) -> (B, N, 1)
            m = mask.unsqueeze(-1)
            # Sum only non-masked nodes
            agg = (x * m).mean(dim=1, keepdim=True)  # (B, 1, Din)
        else:
            agg = x.mean(dim=1, keepdim=True)  # (B, 1, Din)

        # 3. Project the global aggregation and add via broadcasting
        # linear(agg) is (B, 1, Dout), which broadcasts to (B, N, Dout)
        out += self.linears[1](agg)

        # 4. Apply Bias
        out = out + self.bias[:, 0]

        # 5. Final Masking
        if mask is not None:
            out = out * mask.unsqueeze(-1)

        return out


class DenseGraphConvEdgeToNode(DenseGraphConv):
    def __init__(self, in_channels, out_channels, bias=None, include_self_loops=False):
        n_lin = 5 if include_self_loops else 3
        n_bias = 1
        super().__init__(in_channels, out_channels, n_lin, n_bias, bias)
        self.include_self_loops = include_self_loops

    def equivariant_basis(self, E, mask):  # Taille 5

        if mask is not None:
            mask_edge = mask.unsqueeze(1) * mask.unsqueeze(2)
            E = E * mask_edge.unsqueeze(-1)

        Esum_1 = E.mean(dim=1)

        Esum_2 = E.mean(dim=2)

        Esum_all = E.mean(1).mean(dim=2, keepdim=True).expand_as(Esum_1)

        if self.include_self_loops:
            E_diag = E.diagonal(dim1=1, dim2=2).transpose(1, 2)
            E_diag_sum = E_diag.mean(dim=1, keepdim=True).expand_as(Esum_1)
            return E_diag, Esum_1, Esum_2, Esum_all, E_diag_sum
        else:
            return Esum_1, Esum_2, Esum_all

    # def forward(self, x, adj, mask=None, edge_attrs=None):

    #     out = self.equivariant_basis(edge_attrs, mask)  # Compute the basis part
    #     bias = self.bias[:, 0]
    #     if mask is not None:
    #         bias = bias * mask.unsqueeze(-1)
    #     return sum(self.linears[i](x) for i, x in enumerate(out)) + bias

    def forward(self, x, adj, mask=None, edge_attrs=None):
        # edge_attrs (E): (B, N, N, Din)
        E = edge_attrs
        B, N, _, D = E.shape

        # 1. Apply node masking logic efficiently
        # Instead of E = E * mask_edge (N^2 tensor), we apply mask to the sums (N tensor)
        # However, to ensure the sums don't include "junk" from padding:
        if mask is not None:
            # m: (B, N, 1)
            m = mask.unsqueeze(-1)
            # Zero out contributions from masked columns/rows before summing
            E = E * m.unsqueeze(1)  # Mask columns (B, 1, N, 1) - Still creates a tensor,
            # but we only do it once.

        # 2. Compute Aggregations
        # Sum over rows and columns -> (B, N, Din)
        Esum_1 = E.mean(dim=1)
        Esum_2 = E.mean(dim=2)
        # Global sum -> (B, 1, Din)
        Esum_all = Esum_1.mean(dim=1, keepdim=True)

        # 3. Linear Projections and In-place accumulation
        if self.include_self_loops:
            # According to your original order: E_diag, Esum_1, Esum_2, Esum_all, E_diag_sum
            E_diag = E.diagonal(dim1=1, dim2=2).transpose(1, 2)  # (B, N, Din)
            E_diag_sum = E_diag.mean(dim=1, keepdim=True)  # (B, 1, Din)

            out = self.linears[0](E_diag)  # Op 0
            out += self.linears[1](Esum_1)  # Op 1
            out += self.linears[2](Esum_2)  # Op 2
            out += self.linears[3](Esum_all)  # Op 3 (Broadcasts B,1,D -> B,N,D)
            out += self.linears[4](E_diag_sum)  # Op 4 (Broadcasts B,1,D -> B,N,D)
        else:
            # According to your original order: Esum_1, Esum_2, Esum_all
            out = self.linears[0](Esum_1)
            out += self.linears[1](Esum_2)
            out += self.linears[2](Esum_all)  # Broadcasts

        # 4. Bias and Masking
        out = out + self.bias[:, 0]

        if mask is not None:
            # Final clean up to ensure padding nodes are zero
            out = out * mask.unsqueeze(-1)

        return out


class DenseGraphConvNodeToEdge(DenseGraphConv):
    def __init__(self, in_channels, out_channels, bias=None, include_self_loops=False):
        n_lin = 5 if include_self_loops else 3
        n_bias = 2 if include_self_loops else 1
        super().__init__(in_channels, out_channels, n_lin, n_bias, bias)
        self.include_self_loops = include_self_loops

    def equivariant_basis(self, X, mask=None):
        # Taille 5
        _, N, _ = X.shape
        X_columns = X.unsqueeze(1).expand(-1, N, -1, -1)
        X_rows = X.unsqueeze(2).expand_as(X_columns)
        sum_all = X.mean(dim=1, keepdim=True)
        X_sum = sum_all.unsqueeze(1).expand_as(X_columns)

        if self.include_self_loops:

            X_diag = torch.diag_embed(X.transpose(1, 2)).permute(0, 2, 3, 1)
            X_diag_sum = torch.diag_embed(sum_all.expand_as(X).transpose(1, 2)).permute(0, 2, 3, 1)
            return X_columns, X_rows, X_sum, X_diag, X_diag_sum
        else:
            return X_columns, X_rows, X_sum

    # def forward(self, x, adj, mask=None, edge_attrs=None):

    #     out = self.equivariant_basis(x, mask)  # Compute the basis part
    #     bias = self.bias[:, 0]
    #     if self.include_self_loops:
    #         bias += (torch.ones_like(x[:, :, 0]) - 2 * torch.diag(torch.ones_like(x[:, 0, 0]))).unsqueeze(-1) * self.bias[:, 1]
    #     if mask is not None:
    #         mask_edge = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    #         bias = bias * mask_edge.unsqueeze(-1)  # Es-ce que ça marche pour le deuxième bias?
    #     return sum(self.linears[i](x) for i, x in enumerate(out)) + bias  # Et pour l'autre biais il faut y mettre un pattern un peu plus complexe

    def forward(self, x, adj, mask=None, edge_attrs=None):
        # x shape: (B, N, Din)
        B, N, D = x.shape
        device = x.device

        # 1. Compute Compact Aggregations
        sum_all = x.mean(dim=1, keepdim=True)  # (B, 1, Din)

        # 2. Linear Projections + Broadcasting (Prevents materializing massive tensors)
        # Instead of expanding x and then projecting, we project and let broadcasting expand.

        # Op 0: X_columns (Node features depend on row index)
        out = self.linears[0](x.unsqueeze(2))  # (B, N, 1, Dout)

        # Op 1: X_rows (Node features depend on column index)
        out = out + self.linears[1](x.unsqueeze(1))  # (B, 1, N, Dout)

        # Op 2: X_sum (Global sum repeated everywhere)
        out = out + self.linears[2](sum_all.unsqueeze(1))  # (B, 1, 1, Dout)

        if self.include_self_loops:
            # Op 3: X_diag (Node features only on diagonal)
            # Op 4: X_diag_sum (Global sum only on diagonal)
            proj_diag = self.linears[3](x)  # (B, N, Dout)
            proj_diag_sum = self.linears[4](sum_all)  # (B, 1, Dout)

            # Efficiently add to diagonal using indexing
            idx = torch.arange(N, device=device)
            out[:, idx, idx, :] += proj_diag
            out[:, idx, idx, :] += proj_diag_sum

        # 3. Handle Bias
        # bias[:, 0] is (Dout,)
        final_bias = self.bias[:, 0]

        if self.include_self_loops:
            # bias[:, 1] is (Dout,)
            # The logic: 1 on off-diagonal, -1 on diagonal
            eye = torch.eye(N, device=device).unsqueeze(-1)  # (N, N, 1)
            # (1 - 2*eye) creates the 1/-1 pattern
            final_bias = final_bias + (1.0 - 2.0 * eye) * self.bias[:, 1]

        # 4. Handle Masking
        if mask is not None:
            # mask: (B, N) -> mask_edge: (B, N, N, 1)
            mask_edge = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(-1)
            out = out * mask_edge
            # If bias is not a scalar per channel but a pattern, apply mask to it too
            if isinstance(final_bias, torch.Tensor) and final_bias.ndim > 1:
                final_bias = final_bias * mask_edge

        return out + final_bias


class DenseGraphConvEdgeToEdge(DenseGraphConv):
    def __init__(self, in_channels, out_channels, bias=None, include_self_loops=False):
        n_lin = 15 if include_self_loops else 7
        n_bias = 2 if include_self_loops else 1
        super().__init__(in_channels, out_channels, n_lin, n_bias, bias)
        self.include_self_loops = include_self_loops

    def equivariant_basis(self, E, mask=None):
        # Size 7 or 15

        # Sum over columns (dim 2) -> result depends on rows
        sum_rows = E.mean(dim=2, keepdim=True)  # B x N x 1 x D
        # Sum over rows (dim 1) -> result depends on cols
        sum_cols = E.mean(dim=1, keepdim=True)  # B x 1x N x D
        # Global sum
        sum_all = E.mean(dim=(1, 2), keepdim=True)  # B x 1 x 1 D

        rows_to_rows = sum_rows.expand_as(E)
        cols_to_cols = sum_cols.expand_as(E)
        sumall_to_E = sum_all.expand_as(E)

        # Op 9: Tile sum_rows (N, m, D) along dim 1
        rows_to_cols = sum_rows.transpose(1, 2).expand_as(E)
        cols_to_rows = sum_cols.transpose(1, 2).expand_as(E)

        transpose = torch.transpose(E, 1, 2)

        if self.include_self_loops:
            diag_part = torch.diagonal(E, dim1=1, dim2=2).transpose(1, 2)  # B x N x D
            sum_diag_part = diag_part.mean(dim=1, keepdim=True)  # B x 1 x D

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

    # def forward(self, x, adj, mask=None, edge_attrs=None):

    #     out = self.equivariant_basis(edge_attrs, mask)  # Compute the basis part
    #     bias = self.bias[:, 0]
    #     if self.include_self_loops:
    #         bias += (torch.ones_like(x[:, :, 0]) - 2 * torch.diag(torch.ones_like(x[:, 0, 0]))).unsqueeze(-1) * self.bias[:, 1]
    #     if mask is not None:
    #         bias = bias * mask.unsqueeze(-1).unsqueeze(-1)  # Es-ce que ça marche pour le deuxième bias?
    #     return sum(self.linears[i](x) for i, x in enumerate(out)) + bias  # Et pour l'autre biais il faut y mettre un pattern un peu plus complexe

    def forward(self, x, adj, mask=None, edge_attrs=None):
        E = edge_attrs  # Shape: (B, N, N, D)
        B, N, _, D = E.shape

        # 1. Compute Compact Aggregations
        sum_rows = E.mean(dim=2, keepdim=True)  # (B, N, 1, D)
        sum_cols = E.mean(dim=1, keepdim=True)  # (B, 1, N, D)
        sum_all = E.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1, D)

        # 2. Accumulate results into one tensor (Saves 15x memory)
        # We project the "small" tensors and let broadcasting handle the expansion
        out = self.linears[0](E)  # Op 1: E
        out += self.linears[1](E.transpose(1, 2))  # Op 2: E^T
        out += self.linears[2](sum_rows)  # Op 3: sum_rows (broadcasts to B, N, N, D_out)
        out += self.linears[3](sum_cols)  # Op 4: sum_cols
        out += self.linears[4](sum_rows.transpose(1, 2))  # Op 5: row-to-cols
        out += self.linears[5](sum_cols.transpose(1, 2))  # Op 6: col-to-rows
        out += self.linears[6](sum_all)  # Op 7: sum_all

        if self.include_self_loops:
            # Diagonals: (B, N, D)
            diag_part = torch.diagonal(E, dim1=1, dim2=2).transpose(1, 2)
            sum_diag = diag_part.mean(dim=1, keepdim=True)  # (B, 1, D)

            # Projection of diagonal elements
            proj_diag = self.linears[7](diag_part)  # (B, N, D_out)
            proj_sum_rows_d = self.linears[8](sum_rows.squeeze(2))  # (B, N, D_out)
            proj_sum_cols_d = self.linears[9](sum_cols.squeeze(1))  # (B, N, D_out)
            proj_sum_all_d = self.linears[10](sum_all.squeeze(1))  # (B, 1, D_out)
            proj_diag_sum_d = self.linears[13](sum_diag)  # (B, 1, D_out)

            # Efficient Diagonal Embedding (In-place)
            idx = torch.arange(N, device=E.device)
            out[:, idx, idx, :] += proj_diag
            out[:, idx, idx, :] += proj_sum_rows_d
            out[:, idx, idx, :] += proj_sum_cols_d
            out[:, idx, idx, :] += proj_sum_all_d
            out[:, idx, idx, :] += proj_diag_sum_d

            # Other broadcasted terms
            out += self.linears[11](diag_part.unsqueeze(2))  # diag_to_rows: (B, N, 1, D_out)
            out += self.linears[12](diag_part.unsqueeze(1))  # diag_to_cols: (B, 1, N, D_out)
            out += self.linears[14](sum_diag.unsqueeze(1))  # diag_sum_to_E: (B, 1, 1, D_out)

        # 3. Apply Bias
        # bias shape: (out_channels, n_bias) -> (D_out, 1 or 2)
        final_bias = self.bias[:, 0]  # (D_out)
        if self.include_self_loops:
            # include_self_loops bias logic
            eye = torch.eye(N, device=E.device)
            # Pattern: (1 - 2*I) is 1 on off-diag, -1 on diag
            # Broadcasters: (N, N, 1) * (D_out)
            diag_bias = (1.0 - 2.0 * eye).unsqueeze(-1) * self.bias[:, 1]
            final_bias = final_bias + diag_bias
        if mask is not None:
            edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
            # Expand to (B, N, N, 1) for broadcasting with D_out
            final_bias = final_bias * edge_mask.unsqueeze(-1)
        return out + final_bias
