import torch
import torch.nn as nn
from torch_scatter import scatter_add

bell_numbers = [1, 1, 2, 5, 15, 52, 203]


class GraphConv(nn.Module):
    """Base class for sparse graph convolution layers using PyTorch Geometric format.

    This base class handles linear transformations and bias management for equivariant
    graph convolutions operating on sparse edge_index representations.

    Args:
        in_channels (int): Number of input features
        out_channels (int): Number of output features
        n_linears (int): Number of linear transformation layers (one per basis function)
        n_bias (int): Number of bias terms (1 or 2, where 2 is used for self-loop patterns)
        bias (float, optional): Initial bias value. If None, bias is set to zero and not trainable.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_linears,
        n_bias,
        bias=None,
    ):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linears = nn.ModuleList([
            nn.Linear(self.in_channels, self.out_channels, bias=False)
            for _ in range(n_linears)
        ])
        if bias is not None:
            self.bias = torch.nn.Parameter(bias * torch.ones(out_channels, n_bias))
        else:
            self.register_buffer("bias", torch.zeros(out_channels, n_bias))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        for lin in self.linears:
            lin.reset_parameters()
        if self.bias.requires_grad:
            nn.init.uniform_(self.bias)


class GraphConvNodeToNode(GraphConv):
    """Sparse graph convolution layer: Node features → Node features.

    This layer transforms node features while maintaining permutation equivariance.
    Uses 2 equivariant basis functions:
    1. Identity (node features)
    2. Global aggregation (sum of all nodes, broadcast to each node)

    Args:
        in_channels (int): Number of input node features
        out_channels (int): Number of output node features
        bias (float, optional): Initial bias value

    Example:
        >>> layer = GraphConvNodeToNode(5, 8)
        >>> x = torch.randn(10, 5)  # 10 nodes, 5 features
        >>> edge_index = torch.randint(0, 10, (2, 20))
        >>> out = layer(x, edge_index)  # shape: (10, 8)
    """
    def __init__(self, in_channels, out_channels, bias=None):
        super().__init__(in_channels, out_channels, 2, 1, bias)

    def equivariant_basis(self, x, edge_index, batch=None):
        """Compute equivariant basis functions for node-to-node transformation.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges] (not used but kept for API consistency)
            batch: Batch vector [num_nodes] indicating which graph each node belongs to

        Returns:
            Tuple of 2 basis tensors, each with shape [num_nodes, in_channels]
        """
        num_nodes = x.size(0)

        # Basis 1: Identity
        basis_1 = x

        # Basis 2: Global aggregation per graph
        if batch is None:
            # Single graph: sum all nodes and broadcast
            global_agg = x.sum(dim=0, keepdim=True)
            basis_2 = global_agg.expand(num_nodes, -1)
        else:
            # Batched graphs: sum per graph, then broadcast within each graph
            num_graphs = int(batch.max().item()) + 1
            global_agg = scatter_add(x, batch, dim=0, dim_size=num_graphs)
            basis_2 = global_agg[batch]

        return basis_1, basis_2

    def forward(self, x, edge_index, batch=None):
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Optional batch vector [num_nodes]

        Returns:
            Transformed node features [num_nodes, out_channels]
        """
        basis = self.equivariant_basis(x, edge_index, batch)
        out = sum(self.linears[i](b) for i, b in enumerate(basis))
        bias = self.bias[:, 0]
        return out + bias


class GraphConvEdgeToNode(GraphConv):
    """Sparse graph convolution layer: Edge features → Node features.

    This layer aggregates edge features to produce node features while maintaining
    permutation equivariance. Uses 3 or 5 equivariant basis functions depending on
    whether self-loops are included.

    Without self-loops (3 basis functions):
    1. Incoming edge aggregation (sum edges targeting each node)
    2. Outgoing edge aggregation (sum edges sourcing from each node)
    3. Global edge aggregation (sum all edges, broadcast to nodes)

    With self-loops (5 basis functions):
    4. Self-loop features (diagonal elements)
    5. Global self-loop aggregation

    Args:
        in_channels (int): Number of input edge features
        out_channels (int): Number of output node features
        bias (float, optional): Initial bias value
        include_self_loops (bool): Whether to include self-loop basis functions

    Example:
        >>> layer = GraphConvEdgeToNode(3, 8, include_self_loops=False)
        >>> edge_attr = torch.randn(20, 3)  # 20 edges, 3 features
        >>> edge_index = torch.randint(0, 10, (2, 20))
        >>> out = layer(edge_attr, edge_index, num_nodes=10)  # shape: (10, 8)
    """
    def __init__(self, in_channels, out_channels, bias=None, include_self_loops=False):
        n_lin = 5 if include_self_loops else 3
        n_bias = 1
        super().__init__(in_channels, out_channels, n_lin, n_bias, bias)
        self.include_self_loops = include_self_loops

    def equivariant_basis(self, edge_attr, edge_index, num_nodes=None, batch=None):
        """Compute equivariant basis functions for edge-to-node transformation.

        Args:
            edge_attr: Edge features [num_edges, in_channels]
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes (required if some nodes have no edges)
            batch: Batch vector [num_nodes] for batched graphs

        Returns:
            Tuple of 3 or 5 basis tensors, each with shape [num_nodes, in_channels]
        """
        if num_nodes is None:
            num_nodes = int(edge_index.max().item()) + 1

        src, tgt = edge_index[0], edge_index[1]

        # Basis 1: Incoming edge aggregation (sum edges where node is target)
        basis_incoming = scatter_add(edge_attr, tgt, dim=0, dim_size=num_nodes)

        # Basis 2: Outgoing edge aggregation (sum edges where node is source)
        basis_outgoing = scatter_add(edge_attr, src, dim=0, dim_size=num_nodes)

        # Basis 3: Global edge aggregation
        if batch is None:
            global_edge_agg = edge_attr.sum(dim=0, keepdim=True)
            basis_global = global_edge_agg.expand(num_nodes, -1)
        else:
            # Edge batch: edges inherit batch from source node
            edge_batch = batch[src]
            num_graphs = int(batch.max().item()) + 1
            global_edge_agg = scatter_add(edge_attr, edge_batch, dim=0, dim_size=num_graphs)
            basis_global = global_edge_agg[batch]

        if not self.include_self_loops:
            return basis_incoming, basis_outgoing, basis_global

        # With self-loops: extract diagonal elements
        self_loop_mask = (src == tgt)

        # Basis 4: Self-loop features (0 for nodes without self-loops)
        basis_self_loops = torch.zeros(num_nodes, edge_attr.size(1),
                                       device=edge_attr.device, dtype=edge_attr.dtype)
        if self_loop_mask.any():
            basis_self_loops[src[self_loop_mask]] = edge_attr[self_loop_mask]

        # Basis 5: Sum of all self-loop features, broadcast to all nodes
        if batch is None:
            if self_loop_mask.any():
                self_loop_sum = edge_attr[self_loop_mask].sum(dim=0, keepdim=True)
            else:
                self_loop_sum = torch.zeros(1, edge_attr.size(1),
                                           device=edge_attr.device, dtype=edge_attr.dtype)
            basis_self_loop_global = self_loop_sum.expand(num_nodes, -1)
        else:
            # Batched self-loop aggregation
            if self_loop_mask.any():
                self_loop_edge_attr = edge_attr[self_loop_mask]
                self_loop_batch = edge_batch[self_loop_mask]
                num_graphs = int(batch.max().item()) + 1
                self_loop_global = scatter_add(self_loop_edge_attr, self_loop_batch,
                                              dim=0, dim_size=num_graphs)
            else:
                num_graphs = int(batch.max().item()) + 1
                self_loop_global = torch.zeros(num_graphs, edge_attr.size(1),
                                              device=edge_attr.device, dtype=edge_attr.dtype)
            basis_self_loop_global = self_loop_global[batch]

        return basis_incoming, basis_outgoing, basis_global, basis_self_loops, basis_self_loop_global

    def forward(self, edge_attr, edge_index, num_nodes=None, batch=None):
        """Forward pass.

        Args:
            edge_attr: Edge features [num_edges, in_channels]
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            batch: Optional batch vector [num_nodes]

        Returns:
            Node features [num_nodes, out_channels]
        """
        basis = self.equivariant_basis(edge_attr, edge_index, num_nodes, batch)
        out = sum(self.linears[i](b) for i, b in enumerate(basis))
        bias = self.bias[:, 0]
        return out + bias


class GraphConvNodeToEdge(GraphConv):
    """Sparse graph convolution layer: Node features → Edge features.

    This layer broadcasts node features to edges while maintaining permutation equivariance.
    Uses 3 or 5 equivariant basis functions depending on whether self-loops are included.

    Without self-loops (3 basis functions):
    1. Source node features (broadcast to outgoing edges)
    2. Target node features (broadcast to incoming edges)
    3. Global node aggregation (sum all nodes, broadcast to all edges)

    With self-loops (5 basis functions):
    4. Diagonal pattern (node features on self-loops only)
    5. Diagonal global (global sum on self-loops only)

    Args:
        in_channels (int): Number of input node features
        out_channels (int): Number of output edge features
        bias (float, optional): Initial bias value
        include_self_loops (bool): Whether to include self-loop basis functions

    Example:
        >>> layer = GraphConvNodeToEdge(5, 8, include_self_loops=False)
        >>> x = torch.randn(10, 5)  # 10 nodes, 5 features
        >>> edge_index = torch.randint(0, 10, (2, 20))
        >>> out = layer(x, edge_index)  # shape: (20, 8)
    """
    def __init__(self, in_channels, out_channels, bias=None, include_self_loops=False):
        n_lin = 5 if include_self_loops else 3
        n_bias = 2 if include_self_loops else 1
        super().__init__(in_channels, out_channels, n_lin, n_bias, bias)
        self.include_self_loops = include_self_loops

    def equivariant_basis(self, x, edge_index, batch=None):
        """Compute equivariant basis functions for node-to-edge transformation.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector [num_nodes] for batched graphs

        Returns:
            Tuple of 3 or 5 basis tensors, each with shape [num_edges, in_channels]
        """
        src, tgt = edge_index[0], edge_index[1]
        num_edges = edge_index.size(1)

        # Basis 1: Source node features (broadcast to edges)
        basis_source = x[src]

        # Basis 2: Target node features (broadcast to edges)
        basis_target = x[tgt]

        # Basis 3: Global node aggregation (broadcast to all edges)
        if batch is None:
            global_node_agg = x.sum(dim=0, keepdim=True)
            basis_global = global_node_agg.expand(num_edges, -1)
        else:
            # Aggregate nodes per graph
            num_graphs = int(batch.max().item()) + 1
            global_node_agg = scatter_add(x, batch, dim=0, dim_size=num_graphs)
            edge_batch = batch[src]  # Edges inherit graph from source
            basis_global = global_node_agg[edge_batch]

        if not self.include_self_loops:
            return basis_source, basis_target, basis_global

        # With self-loops: create diagonal patterns
        self_loop_mask = (src == tgt)

        # Basis 4: For self-loop edges, take node feature; for others, zero
        basis_diag = torch.zeros_like(basis_source)
        if self_loop_mask.any():
            basis_diag[self_loop_mask] = x[src[self_loop_mask]]

        # Basis 5: Global sum broadcast to self-loops only
        basis_diag_global = torch.zeros_like(basis_source)
        if self_loop_mask.any():
            if batch is None:
                global_sum = x.sum(dim=0, keepdim=True)
                basis_diag_global[self_loop_mask] = global_sum.expand(self_loop_mask.sum(), -1)
            else:
                num_graphs = int(batch.max().item()) + 1
                global_sum = scatter_add(x, batch, dim=0, dim_size=num_graphs)
                edge_batch_self = batch[src[self_loop_mask]]
                basis_diag_global[self_loop_mask] = global_sum[edge_batch_self]

        return basis_source, basis_target, basis_global, basis_diag, basis_diag_global

    def forward(self, x, edge_index, batch=None):
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Optional batch vector [num_nodes]

        Returns:
            Edge features [num_edges, out_channels]
        """
        basis = self.equivariant_basis(x, edge_index, batch)
        out = sum(self.linears[i](b) for i, b in enumerate(basis))

        # Bias handling with diagonal pattern for self-loops
        bias = self.bias[:, 0]
        if self.include_self_loops and self.bias.shape[1] == 2:
            # Pattern: +1 for non-diagonal, -1 for diagonal edges
            src, tgt = edge_index[0], edge_index[1]
            self_loop_mask = (src == tgt)
            bias_pattern = torch.ones(edge_index.size(1), 1, device=bias.device, dtype=bias.dtype)
            bias_pattern[self_loop_mask] = -1
            bias = bias + bias_pattern * self.bias[:, 1]

        return out + bias


class GraphConvEdgeToEdge(GraphConv):
    """Sparse graph convolution layer: Edge features → Edge features.

    This layer transforms edge features while maintaining permutation equivariance.
    Uses 7 or 15 equivariant basis functions depending on whether self-loops are included.

    Without self-loops (7 basis functions):
    1. Identity (edge features)
    2. Transpose (for edge (i,j), get features of edge (j,i))
    3. Row aggregation (for edge (i,j), sum all edges (i,*))
    4. Column aggregation (for edge (i,j), sum all edges (*,j))
    5. Row-to-column (row sums broadcast via target)
    6. Column-to-row (column sums broadcast via source)
    7. Global aggregation (sum all edges, broadcast)

    With self-loops (15 basis functions):
    8-15. Various diagonal operations (self-loop patterns)

    Args:
        in_channels (int): Number of input edge features
        out_channels (int): Number of output edge features
        bias (float, optional): Initial bias value
        include_self_loops (bool): Whether to include self-loop basis functions

    Example:
        >>> layer = GraphConvEdgeToEdge(3, 8, include_self_loops=False)
        >>> edge_attr = torch.randn(20, 3)  # 20 edges, 3 features
        >>> edge_index = torch.randint(0, 10, (2, 20))
        >>> out = layer(edge_attr, edge_index, num_nodes=10)  # shape: (20, 8)
    """
    def __init__(self, in_channels, out_channels, bias=None, include_self_loops=False):
        n_lin = 15 if include_self_loops else 7
        n_bias = 2 if include_self_loops else 1
        super().__init__(in_channels, out_channels, n_lin, n_bias, bias)
        self.include_self_loops = include_self_loops

    def equivariant_basis(self, edge_attr, edge_index, num_nodes=None, batch=None):
        """Compute equivariant basis functions for edge-to-edge transformation.

        Args:
            edge_attr: Edge features [num_edges, in_channels]
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes (needed for aggregations)
            batch: Batch vector [num_nodes] for batched graphs

        Returns:
            Tuple of 7 or 15 basis tensors, each with shape [num_edges, in_channels]
        """
        if num_nodes is None:
            num_nodes = int(edge_index.max().item()) + 1

        src, tgt = edge_index[0], edge_index[1]
        num_edges = edge_index.size(1)

        # Basis 1: Identity
        basis_identity = edge_attr

        # Basis 2: Transpose - for edge (i,j), get features of edge (j,i)
        # Build edge -> index mapping
        edge_to_idx = {}
        for idx in range(num_edges):
            edge_to_idx[(src[idx].item(), tgt[idx].item())] = idx

        basis_transpose = torch.zeros_like(edge_attr)
        for idx in range(num_edges):
            s, t = src[idx].item(), tgt[idx].item()
            reverse_idx = edge_to_idx.get((t, s))
            if reverse_idx is not None:
                basis_transpose[idx] = edge_attr[reverse_idx]

        # Basis 3: Row aggregation - for edge (i,j), sum all edges (i,*)
        row_agg = scatter_add(edge_attr, src, dim=0, dim_size=num_nodes)
        basis_row = row_agg[src]

        # Basis 4: Column aggregation - for edge (i,j), sum all edges (*,j)
        col_agg = scatter_add(edge_attr, tgt, dim=0, dim_size=num_nodes)
        basis_col = col_agg[tgt]

        # Basis 5: Row-to-column - row sums broadcast via target
        basis_row_to_col = row_agg[tgt]

        # Basis 6: Column-to-row - column sums broadcast via source
        basis_col_to_row = col_agg[src]

        # Basis 7: Global aggregation
        if batch is None:
            global_agg = edge_attr.sum(dim=0, keepdim=True)
            basis_global = global_agg.expand(num_edges, -1)
        else:
            edge_batch = batch[src]
            num_graphs = int(batch.max().item()) + 1
            global_agg = scatter_add(edge_attr, edge_batch, dim=0, dim_size=num_graphs)
            basis_global = global_agg[edge_batch]

        if not self.include_self_loops:
            return (basis_identity, basis_transpose, basis_row, basis_col,
                   basis_row_to_col, basis_col_to_row, basis_global)

        # With self-loops (8 additional basis functions)
        self_loop_mask = (src == tgt)

        # Basis 8: Diagonal identity - self-loop features only
        basis_diag = torch.zeros_like(edge_attr)
        if self_loop_mask.any():
            basis_diag[self_loop_mask] = edge_attr[self_loop_mask]

        # Extract self-loop features and aggregate
        if self_loop_mask.any():
            self_loop_attr = edge_attr[self_loop_mask]
            self_loop_nodes = src[self_loop_mask]
        else:
            self_loop_attr = torch.zeros(0, edge_attr.size(1),
                                        device=edge_attr.device, dtype=edge_attr.dtype)
            self_loop_nodes = torch.zeros(0, dtype=torch.long, device=src.device)

        # Basis 9: Row aggregation to diagonal
        basis_row_to_diag = torch.zeros_like(edge_attr)
        if self_loop_mask.any():
            basis_row_to_diag[self_loop_mask] = row_agg[self_loop_nodes]

        # Basis 10: Column aggregation to diagonal
        basis_col_to_diag = torch.zeros_like(edge_attr)
        if self_loop_mask.any():
            basis_col_to_diag[self_loop_mask] = col_agg[self_loop_nodes]

        # Basis 11: Global to diagonal
        basis_global_to_diag = torch.zeros_like(edge_attr)
        if self_loop_mask.any():
            if batch is None:
                basis_global_to_diag[self_loop_mask] = global_agg.expand(self_loop_mask.sum(), -1)
            else:
                edge_batch_self = batch[self_loop_nodes]
                basis_global_to_diag[self_loop_mask] = global_agg[edge_batch_self]

        # Basis 12: Diagonal to rows - broadcast self-loop features to row
        diag_agg = torch.zeros(num_nodes, edge_attr.size(1),
                              device=edge_attr.device, dtype=edge_attr.dtype)
        if self_loop_mask.any():
            diag_agg[self_loop_nodes] = self_loop_attr
        basis_diag_to_row = diag_agg[src]

        # Basis 13: Diagonal to columns
        basis_diag_to_col = diag_agg[tgt]

        # Basis 14: Diagonal sum to diagonal
        if batch is None:
            if self_loop_mask.any():
                diag_sum = self_loop_attr.sum(dim=0, keepdim=True)
            else:
                diag_sum = torch.zeros(1, edge_attr.size(1),
                                      device=edge_attr.device, dtype=edge_attr.dtype)
            basis_diag_sum_to_diag = torch.zeros_like(edge_attr)
            if self_loop_mask.any():
                basis_diag_sum_to_diag[self_loop_mask] = diag_sum.expand(self_loop_mask.sum(), -1)
        else:
            if self_loop_mask.any():
                self_loop_batch = batch[self_loop_nodes]
                num_graphs = int(batch.max().item()) + 1
                diag_sum = scatter_add(self_loop_attr, self_loop_batch, dim=0, dim_size=num_graphs)
            else:
                num_graphs = int(batch.max().item()) + 1
                diag_sum = torch.zeros(num_graphs, edge_attr.size(1),
                                      device=edge_attr.device, dtype=edge_attr.dtype)
            basis_diag_sum_to_diag = torch.zeros_like(edge_attr)
            if self_loop_mask.any():
                basis_diag_sum_to_diag[self_loop_mask] = diag_sum[batch[self_loop_nodes]]

        # Basis 15: Diagonal sum to all edges
        if batch is None:
            basis_diag_sum_global = diag_sum.expand(num_edges, -1)
        else:
            edge_batch = batch[src]
            basis_diag_sum_global = diag_sum[edge_batch]

        return (basis_identity, basis_transpose, basis_row, basis_col,
               basis_row_to_col, basis_col_to_row, basis_global,
               basis_diag, basis_row_to_diag, basis_col_to_diag,
               basis_global_to_diag, basis_diag_to_row, basis_diag_to_col,
               basis_diag_sum_to_diag, basis_diag_sum_global)

    def forward(self, edge_attr, edge_index, num_nodes=None, batch=None):
        """Forward pass.

        Args:
            edge_attr: Edge features [num_edges, in_channels]
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            batch: Optional batch vector [num_nodes]

        Returns:
            Edge features [num_edges, out_channels]
        """
        basis = self.equivariant_basis(edge_attr, edge_index, num_nodes, batch)
        out = sum(self.linears[i](b) for i, b in enumerate(basis))

        # Bias handling with diagonal pattern for self-loops
        bias = self.bias[:, 0]
        if self.include_self_loops and self.bias.shape[1] == 2:
            # Pattern: +1 for non-diagonal, -1 for diagonal edges
            src, tgt = edge_index[0], edge_index[1]
            self_loop_mask = (src == tgt)
            bias_pattern = torch.ones(edge_index.size(1), 1, device=bias.device, dtype=bias.dtype)
            bias_pattern[self_loop_mask] = -1
            bias = bias + bias_pattern * self.bias[:, 1]

        return out + bias
