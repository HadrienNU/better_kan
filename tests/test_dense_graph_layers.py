"""Comprehensive tests for dense graph convolution layers.

This module tests:
1. Basic functionality and output shapes for all layer types
2. Permutation equivariance for all layer types
3. Batching support with mask vectors
4. Self-loops handling
"""

import torch
import pytest
from better_kan.dense_graph_layers import (
    DenseGraphConvNodeToNode,
    DenseGraphConvEdgeToNode,
    DenseGraphConvNodeToEdge,
    DenseGraphConvEdgeToEdge,
)


# ========================================
# Utility Functions
# ========================================


def create_random_dense_graph(num_nodes, num_edges, node_features, edge_features=None):
    """Create a random dense graph.

    Args:
        num_nodes: Number of nodes
        num_edges: Number of edges (ignored for dense)
        node_features: Number of node features
        edge_features: Number of edge features (None if not needed)

    Returns:
        x, adj, edge_attr (or None if edge_features is None)
    """
    x = torch.randn(num_nodes, node_features)
    adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    # Make adjacency matrix symmetric and add self-loops
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.transpose(0, 1)
    adj.fill_diagonal_(1.0)

    edge_attr = torch.randn(1, num_nodes, num_nodes, edge_features) if edge_features is not None else None
    return x, adj.unsqueeze(0), edge_attr


def create_batched_dense_graph(num_graphs, nodes_per_graph, node_features, edge_features=None):
    """Create a batched dense graph with mask vectors.

    Args:
        num_graphs: Number of graphs in the batch
        nodes_per_graph: List of node counts per graph
        node_features: Number of node features
        edge_features: Number of edge features (None if not needed)

    Returns:
        x, adj, mask, edge_attr (or None)
    """
    x_list = []
    adj_list = []
    edge_attr_list = []
    mask_list = []

    node_offset = 0
    for g in range(num_graphs):
        n_nodes = nodes_per_graph[g]

        # Create graph
        x_g = torch.randn(n_nodes, node_features)
        adj_g = torch.randint(0, 2, (n_nodes, n_nodes)).float()
        adj_g = torch.triu(adj_g, diagonal=1)
        adj_g = adj_g + adj_g.transpose(0, 1)
        adj_g.fill_diagonal_(1.0)

        # Create mask
        mask_g = torch.ones(n_nodes, dtype=torch.bool)

        x_list.append(x_g)
        adj_list.append(adj_g)
        mask_list.append(mask_g)

        if edge_features is not None:
            edge_attr_g = torch.randn(n_nodes, n_nodes, edge_features)
            edge_attr_list.append(edge_attr_g)

        node_offset += n_nodes

    # Pad to batch size
    max_nodes = max(nodes_per_graph)
    x = torch.zeros(num_graphs, max_nodes, node_features)
    adj = torch.zeros(num_graphs, max_nodes, max_nodes)
    mask = torch.zeros(num_graphs, max_nodes, dtype=torch.bool)

    if edge_features is not None:
        edge_attr = torch.zeros(num_graphs, max_nodes, max_nodes, edge_features)
    else:
        edge_attr = None

    for g in range(num_graphs):
        n_nodes = nodes_per_graph[g]
        x[g, :n_nodes] = x_list[g]
        adj[g, :n_nodes, :n_nodes] = adj_list[g]
        mask[g, :n_nodes] = mask_list[g]

        if edge_features is not None:
            edge_attr[g, :n_nodes, :n_nodes] = edge_attr_list[g]

    return x, adj, mask, edge_attr


def permute_dense_graph(x, adj, edge_attr, mask, perm):
    """Apply node permutation to dense graph.

    Args:
        x: Node features [num_nodes, features]
        adj: Adjacency matrix [num_nodes, num_nodes]
        edge_attr: Edge attributes [num_nodes, num_nodes, features]
        mask: Node mask [num_nodes]
        perm: Permutation

    Returns:
        x_perm, adj_perm, edge_attr_perm, mask_perm
    """
    # Permute node features
    x_perm = x[:, perm]

    # Permute adjacency matrix
    P = torch.eye(adj.size(1))[perm]
    adj_perm = P @ adj @ P.t()

    # Permute edge attributes if not None
    edge_attr_perm = None
    if edge_attr is not None:
        edge_attr_perm = edge_attr[perm][:, perm]

    # Permute mask
    mask_perm = mask[:, perm]

    return x_perm, adj_perm, edge_attr_perm, mask_perm


# ========================================
# Test: Basic Functionality
# ========================================


def test_dense_node_to_node_basic():
    """Test DenseGraphConvNodeToNode basic functionality."""
    num_graphs = 1
    nodes_per_graph = [10]
    in_channels, out_channels = 5, 8

    x, adj, mask, _ = create_batched_dense_graph(num_graphs, nodes_per_graph, in_channels)

    layer = DenseGraphConvNodeToNode(in_channels, out_channels, bias=1.0)
    out = layer(x, adj, mask=mask)

    total_nodes = sum(nodes_per_graph)
    assert out.shape == (1, total_nodes, out_channels), f"Expected shape (1, {total_nodes}, {out_channels}), got {out.shape}"
    print("✓ DenseGraphConvNodeToNode basic test passed")


def test_dense_edge_to_node_basic():
    """Test DenseGraphConvEdgeToNode basic functionality."""
    num_graphs = 1
    nodes_per_graph = [10]
    edge_features, out_channels = 3, 8

    _, adj, mask, edge_attr = create_batched_dense_graph(num_graphs, nodes_per_graph, 0, edge_features)

    # Test without self-loops
    layer = DenseGraphConvEdgeToNode(edge_features, out_channels, bias=1.0, include_self_loops=False)
    out = layer(None, adj, mask=mask, edge_attrs=edge_attr)
    total_nodes = sum(nodes_per_graph)
    assert out.shape == (1, total_nodes, out_channels), f"Expected shape (1, {total_nodes}, {out_channels}), got {out.shape}"

    # Test with self-loops
    layer_sl = DenseGraphConvEdgeToNode(edge_features, out_channels, bias=1.0, include_self_loops=True)
    out_sl = layer_sl(None, adj, mask=mask, edge_attrs=edge_attr)
    assert out_sl.shape == (1, total_nodes, out_channels)

    print("✓ DenseGraphConvEdgeToNode basic test passed")


def test_dense_node_to_edge_basic():
    """Test DenseGraphConvNodeToEdge basic functionality."""
    num_graphs = 1
    nodes_per_graph = [10]
    node_features, out_channels = 5, 8

    x, adj, mask, _ = create_batched_dense_graph(num_graphs, nodes_per_graph, node_features)

    # Test without self-loops
    layer = DenseGraphConvNodeToEdge(node_features, out_channels, bias=1.0, include_self_loops=False)
    out = layer(x, adj, mask=mask)
    max_nodes = max(nodes_per_graph)
    assert out.shape == (1, max_nodes, max_nodes, out_channels), f"Expected shape (1, {max_nodes}, {max_nodes}, {out_channels}), got {out.shape}"

    # Test with self-loops
    layer_sl = DenseGraphConvNodeToEdge(node_features, out_channels, bias=1.0, include_self_loops=True)
    out_sl = layer_sl(x, adj, mask=mask)
    assert out_sl.shape == (1, max_nodes, max_nodes, out_channels)

    print("✓ DenseGraphConvNodeToEdge basic test passed")


def test_dense_edge_to_edge_basic():
    """Test DenseGraphConvEdgeToEdge basic functionality."""
    num_graphs = 1
    nodes_per_graph = [10]
    edge_features, out_channels = 3, 8

    _, adj, mask, edge_attr = create_batched_dense_graph(num_graphs, nodes_per_graph, 0, edge_features)

    # Test without self-loops
    layer = DenseGraphConvEdgeToEdge(edge_features, out_channels, bias=1.0, include_self_loops=False)
    out = layer(None, adj, mask=mask, edge_attrs=edge_attr)
    max_nodes = max(nodes_per_graph)
    assert out.shape == (1, max_nodes, max_nodes, out_channels), f"Expected shape (1, {max_nodes}, {max_nodes}, {out_channels}), got {out.shape}"

    # Test with self-loops
    layer_sl = DenseGraphConvEdgeToEdge(edge_features, out_channels, bias=1.0, include_self_loops=True)
    out_sl = layer_sl(None, adj, mask=mask, edge_attrs=edge_attr)
    assert out_sl.shape == (1, max_nodes, max_nodes, out_channels)

    print("✓ DenseGraphConvEdgeToEdge basic test passed")


# ========================================
# Test: Permutation Equivariance
# ========================================


def test_dense_node_to_node_equivariance():
    """Test DenseGraphConvNodeToNode permutation equivariance."""
    num_graphs = 1
    nodes_per_graph = [10]
    in_channels, out_channels = 5, 8

    x, adj, mask, _ = create_batched_dense_graph(num_graphs, nodes_per_graph, in_channels)

    layer = DenseGraphConvNodeToNode(in_channels, out_channels, bias=1.0)

    # Forward on original graph
    out1 = layer(x, adj, mask=mask)

    # Permute graph
    num_nodes = nodes_per_graph[0]
    perm = torch.randperm(num_nodes)
    x_perm, adj_perm, _, mask_perm = permute_dense_graph(x, adj, None, mask, perm)
    out2 = layer(x_perm, adj_perm, mask=mask_perm)

    # Check equivariance: out1[perm] should equal out2
    out1_perm = out1[:, perm]
    diff = (out1_perm - out2).abs().max()

    assert diff < 1e-5, f"Equivariance failed with max diff: {diff.item()}"
    print(f"✓ DenseGraphConvNodeToNode equivariance test passed (max diff: {diff.item():.2e})")


def test_dense_edge_to_node_equivariance():
    """Test DenseGraphConvEdgeToNode permutation equivariance."""
    num_graphs = 1
    nodes_per_graph = [10]
    edge_features, out_channels = 3, 8

    x, adj, mask, edge_attr = create_batched_dense_graph(num_graphs, nodes_per_graph, 5, edge_features)

    # Test both without and with self-loops
    for include_self_loops in [False, True]:
        layer = DenseGraphConvEdgeToNode(edge_features, out_channels, bias=1.0, include_self_loops=include_self_loops)

        # Forward on original graph
        out1 = layer(None, adj, mask=mask, edge_attrs=edge_attr)

        # Permute graph
        num_nodes = nodes_per_graph[0]
        perm = torch.randperm(num_nodes)
        _, adj_perm, edge_attr_perm, mask_perm = permute_dense_graph(x, adj, edge_attr, mask, perm)
        out2 = layer(None, adj_perm, mask=mask_perm, edge_attrs=edge_attr_perm)

        # Check equivariance: out1[perm] should equal out2
        out1_perm = out1[:, perm]
        diff = (out1_perm - out2).abs().max()

        assert diff < 1e-5, f"Equivariance failed (self_loops={include_self_loops}): {diff.item()}"

    print(f"✓ DenseGraphConvEdgeToNode equivariance test passed")


def test_dense_node_to_edge_equivariance():
    """Test DenseGraphConvNodeToEdge permutation equivariance."""
    num_graphs = 1
    nodes_per_graph = [10]
    node_features, out_channels = 5, 8

    x, adj, mask, _ = create_batched_dense_graph(num_graphs, nodes_per_graph, node_features)
    x = x[0]  # Get single graph from batch
    adj = adj[0]  # Get single graph from batch
    mask = mask[0]  # Get single graph from batch

    # Test both without and with self-loops
    for include_self_loops in [False, True]:
        layer = DenseGraphConvNodeToEdge(node_features, out_channels, bias=1.0, include_self_loops=include_self_loops)

        # Forward on original graph
        out1 = layer(x, adj, mask=mask)

        # Permute graph
        num_nodes = nodes_per_graph[0]
        perm = torch.randperm(num_nodes)
        x_perm, adj_perm, _, mask_perm = permute_dense_graph(x, adj, None, mask, perm)
        out2 = layer(x_perm, adj_perm, mask=mask_perm)

        # For node-to-edge: the output should be equivariant under permutation
        diff = (out1 - out2).abs().max()
        assert diff < 1e-5, f"Equivariance failed (self_loops={include_self_loops}): {diff.item()}"

    print(f"✓ DenseGraphConvNodeToEdge equivariance test passed")


def test_dense_edge_to_edge_equivariance():
    """Test DenseGraphConvEdgeToEdge permutation equivariance."""
    num_graphs = 1
    nodes_per_graph = [10]
    edge_features, out_channels = 3, 8

    x, adj, mask, edge_attr = create_batched_dense_graph(num_graphs, nodes_per_graph, 5, edge_features)
    x = x[0]  # Get single graph from batch
    adj = adj[0]  # Get single graph from batch
    mask = mask[0]  # Get single graph from batch
    edge_attr = edge_attr[0] if edge_attr is not None else None

    # Test without self-loops only (with self-loops is more complex)
    layer = DenseGraphConvEdgeToEdge(edge_features, out_channels, bias=1.0, include_self_loops=False)

    # Forward on original graph
    out1 = layer(None, adj, mask=mask, edge_attrs=edge_attr)

    # Permute graph
    num_nodes = nodes_per_graph[0]
    perm = torch.randperm(num_nodes)
    _, adj_perm, edge_attr_perm, mask_perm = permute_dense_graph(x, adj, edge_attr, mask, perm)
    out2 = layer(None, adj_perm, mask=mask_perm, edge_attrs=edge_attr_perm)

    # For edge-to-edge, the output features should be equivariant under permutation
    diff = (out1 - out2).abs().max()
    assert diff < 1e-5, f"Equivariance failed: {diff.item()}"

    print(f"✓ DenseGraphConvEdgeToEdge equivariance test passed (max diff: {diff.item():.2e})")


# ========================================
# Test: Self-Loops
# ========================================


def test_dense_self_loops_handling():
    """Test self-loops handling in layers that support it."""
    num_nodes = 10
    node_features, edge_features = 5, 3
    out_channels = 8

    # Create graph with explicit self-loops (diagonal elements)
    adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.transpose(0, 1)
    adj.fill_diagonal_(1.0)  # Add self-loops

    x = torch.randn(num_nodes, node_features)
    edge_attr = torch.randn(num_nodes, num_nodes, edge_features)

    # Test EdgeToNode with self-loops
    layer_e2n = DenseGraphConvEdgeToNode(edge_features, out_channels, include_self_loops=True)
    out_e2n = layer_e2n(None, adj, edge_attrs=edge_attr)
    assert out_e2n.shape == (num_nodes, out_channels)

    # Test NodeToEdge with self-loops
    layer_n2e = DenseGraphConvNodeToEdge(node_features, out_channels, include_self_loops=True)
    out_n2e = layer_n2e(x, adj)
    assert out_n2e.shape == (num_nodes, num_nodes, out_channels)

    # Test EdgeToEdge with self-loops
    layer_e2e = DenseGraphConvEdgeToEdge(edge_features, out_channels, include_self_loops=True)
    out_e2e = layer_e2e(None, adj, edge_attrs=edge_attr)
    assert out_e2e.shape == (num_nodes, num_nodes, out_channels)

    print("✓ Dense self-loops handling test passed")


# ========================================
# Run All Tests
# ========================================


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Dense Graph Layer Tests")
    print("=" * 60 + "\n")

    print("--- Basic Functionality Tests ---")
    test_dense_node_to_node_basic()
    test_dense_edge_to_node_basic()
    test_dense_node_to_edge_basic()
    test_dense_edge_to_edge_basic()

    print("\n--- Equivariance Tests ---")
    test_dense_node_to_node_equivariance()
    test_dense_edge_to_node_equivariance()
    test_dense_node_to_edge_equivariance()
    test_dense_edge_to_edge_equivariance()

    print("\n--- Self-Loops Tests ---")
    test_dense_self_loops_handling()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
