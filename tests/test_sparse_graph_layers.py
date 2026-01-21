"""Comprehensive tests for sparse graph convolution layers.

This module tests:
1. Basic functionality and output shapes for all layer types
2. Permutation equivariance for all layer types
3. Batching support with PyG batch vectors
4. Self-loops handling
"""

import torch
import pytest
from better_kan.graph_layers import (
    GraphConvNodeToNode,
    GraphConvEdgeToNode,
    GraphConvNodeToEdge,
    GraphConvEdgeToEdge,
)


# ========================================
# Utility Functions
# ========================================

def create_random_graph(num_nodes, num_edges, node_features, edge_features=None):
    """Create a random graph in sparse format.

    Args:
        num_nodes: Number of nodes
        num_edges: Number of edges
        node_features: Number of node features
        edge_features: Number of edge features (None if not needed)

    Returns:
        x, edge_index, edge_attr (or None if edge_features is None)
    """
    x = torch.randn(num_nodes, node_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_features) if edge_features is not None else None
    return x, edge_index, edge_attr


def create_batched_graph(num_graphs, nodes_per_graph, edges_per_graph, node_features, edge_features=None):
    """Create a batched graph with PyG batch vector.

    Args:
        num_graphs: Number of graphs in the batch
        nodes_per_graph: List of node counts per graph
        edges_per_graph: List of edge counts per graph
        node_features: Number of node features
        edge_features: Number of edge features (None if not needed)

    Returns:
        x, edge_index, batch, edge_attr (or None)
    """
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_list = []

    node_offset = 0
    for g in range(num_graphs):
        n_nodes = nodes_per_graph[g]
        n_edges = edges_per_graph[g]

        # Create graph
        x_g = torch.randn(n_nodes, node_features)
        edge_index_g = torch.randint(0, n_nodes, (2, n_edges)) + node_offset

        x_list.append(x_g)
        edge_index_list.append(edge_index_g)
        batch_list.append(torch.full((n_nodes,), g, dtype=torch.long))

        if edge_features is not None:
            edge_attr_g = torch.randn(n_edges, edge_features)
            edge_attr_list.append(edge_attr_g)

        node_offset += n_nodes

    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    batch = torch.cat(batch_list, dim=0)
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_features is not None else None

    return x, edge_index, batch, edge_attr


def permute_graph(x, edge_index, perm=None):
    """Apply node permutation to graph.

    Args:
        x: Node features [num_nodes, features]
        edge_index: Edge indices [2, num_edges]
        perm: Permutation (if None, generate random)

    Returns:
        x_perm, edge_index_perm, perm, inv_perm
    """
    num_nodes = x.size(0)
    if perm is None:
        perm = torch.randperm(num_nodes)

    # Permute node features
    x_perm = x[perm]

    # Remap edge indices
    inv_perm = torch.argsort(perm)
    edge_index_perm = inv_perm[edge_index]

    return x_perm, edge_index_perm, perm, inv_perm


def permute_edge_features(edge_attr, edge_index, perm):
    """Permute edge features according to node permutation.

    When nodes are permuted, edges follow: edge (i,j) stays as edge (i,j) but the
    indices are remapped. The edge features don't need special permutation - they
    just stay with their edges in the same order since edge_index itself gets permuted.

    Args:
        edge_attr: Edge features [num_edges, features]
        edge_index: Original edge indices [2, num_edges]
        perm: Node permutation

    Returns:
        edge_attr: Same as input (no permutation needed)
    """
    # Edge features stay in the same order as the edges in edge_index
    # When we permute edge_index, the edge_attr should stay aligned with it
    return edge_attr


# ========================================
# Test: Basic Functionality
# ========================================

def test_node_to_node_basic():
    """Test GraphConvNodeToNode basic functionality."""
    num_nodes, num_edges = 10, 20
    in_channels, out_channels = 5, 8

    x, edge_index, _ = create_random_graph(num_nodes, num_edges, in_channels)

    layer = GraphConvNodeToNode(in_channels, out_channels, bias=1.0)
    out = layer(x, edge_index)

    assert out.shape == (num_nodes, out_channels), f"Expected shape ({num_nodes}, {out_channels}), got {out.shape}"
    print("✓ GraphConvNodeToNode basic test passed")


def test_edge_to_node_basic():
    """Test GraphConvEdgeToNode basic functionality."""
    num_nodes, num_edges = 10, 20
    edge_features, out_channels = 3, 8

    _, edge_index, edge_attr = create_random_graph(num_nodes, num_edges, 0, edge_features)

    # Test without self-loops
    layer = GraphConvEdgeToNode(edge_features, out_channels, bias=1.0, include_self_loops=False)
    out = layer(edge_attr, edge_index, num_nodes=num_nodes)
    assert out.shape == (num_nodes, out_channels), f"Expected shape ({num_nodes}, {out_channels}), got {out.shape}"

    # Test with self-loops
    layer_sl = GraphConvEdgeToNode(edge_features, out_channels, bias=1.0, include_self_loops=True)
    out_sl = layer_sl(edge_attr, edge_index, num_nodes=num_nodes)
    assert out_sl.shape == (num_nodes, out_channels)

    print("✓ GraphConvEdgeToNode basic test passed")


def test_node_to_edge_basic():
    """Test GraphConvNodeToEdge basic functionality."""
    num_nodes, num_edges = 10, 20
    node_features, out_channels = 5, 8

    x, edge_index, _ = create_random_graph(num_nodes, num_edges, node_features)

    # Test without self-loops
    layer = GraphConvNodeToEdge(node_features, out_channels, bias=1.0, include_self_loops=False)
    out = layer(x, edge_index)
    assert out.shape == (num_edges, out_channels), f"Expected shape ({num_edges}, {out_channels}), got {out.shape}"

    # Test with self-loops
    layer_sl = GraphConvNodeToEdge(node_features, out_channels, bias=1.0, include_self_loops=True)
    out_sl = layer_sl(x, edge_index)
    assert out_sl.shape == (num_edges, out_channels)

    print("✓ GraphConvNodeToEdge basic test passed")


def test_edge_to_edge_basic():
    """Test GraphConvEdgeToEdge basic functionality."""
    num_nodes, num_edges = 10, 20
    edge_features, out_channels = 3, 8

    _, edge_index, edge_attr = create_random_graph(num_nodes, num_edges, 0, edge_features)

    # Test without self-loops
    layer = GraphConvEdgeToEdge(edge_features, out_channels, bias=1.0, include_self_loops=False)
    out = layer(edge_attr, edge_index, num_nodes=num_nodes)
    assert out.shape == (num_edges, out_channels), f"Expected shape ({num_edges}, {out_channels}), got {out.shape}"

    # Test with self-loops
    layer_sl = GraphConvEdgeToEdge(edge_features, out_channels, bias=1.0, include_self_loops=True)
    out_sl = layer_sl(edge_attr, edge_index, num_nodes=num_nodes)
    assert out_sl.shape == (num_edges, out_channels)

    print("✓ GraphConvEdgeToEdge basic test passed")


# ========================================
# Test: Permutation Equivariance
# ========================================

def test_node_to_node_equivariance():
    """Test GraphConvNodeToNode permutation equivariance."""
    num_nodes, num_edges = 10, 20
    in_channels, out_channels = 5, 8

    x, edge_index, _ = create_random_graph(num_nodes, num_edges, in_channels)

    layer = GraphConvNodeToNode(in_channels, out_channels, bias=1.0)

    # Forward on original graph
    out1 = layer(x, edge_index)

    # Permute graph
    x_perm, edge_index_perm, perm, _ = permute_graph(x, edge_index)
    out2 = layer(x_perm, edge_index_perm)

    # Check equivariance: out1[perm] should equal out2
    out1_perm = out1[perm]
    diff = (out1_perm - out2).abs().max()

    assert diff < 1e-5, f"Equivariance failed with max diff: {diff.item()}"
    print(f"✓ GraphConvNodeToNode equivariance test passed (max diff: {diff.item():.2e})")


def test_edge_to_node_equivariance():
    """Test GraphConvEdgeToNode permutation equivariance."""
    num_nodes, num_edges = 10, 20
    edge_features, out_channels = 3, 8

    x, edge_index, edge_attr = create_random_graph(num_nodes, num_edges, 5, edge_features)

    # Test both without and with self-loops
    for include_self_loops in [False, True]:
        layer = GraphConvEdgeToNode(edge_features, out_channels, bias=1.0,
                                   include_self_loops=include_self_loops)

        # Forward on original graph
        out1 = layer(edge_attr, edge_index, num_nodes=num_nodes)

        # Permute graph (edge_attr stays aligned with edges - no reordering)
        _, edge_index_perm, perm, _ = permute_graph(x, edge_index)
        out2 = layer(edge_attr, edge_index_perm, num_nodes=num_nodes)

        # Check equivariance: out1[perm] should equal out2
        out1_perm = out1[perm]
        diff = (out1_perm - out2).abs().max()

        assert diff < 1e-5, f"Equivariance failed (self_loops={include_self_loops}): {diff.item()}"

    print(f"✓ GraphConvEdgeToNode equivariance test passed")


def test_node_to_edge_equivariance():
    """Test GraphConvNodeToEdge permutation equivariance."""
    num_nodes, num_edges = 10, 20
    node_features, out_channels = 5, 8

    x, edge_index, _ = create_random_graph(num_nodes, num_edges, node_features)

    # Test both without and with self-loops
    for include_self_loops in [False, True]:
        layer = GraphConvNodeToEdge(node_features, out_channels, bias=1.0,
                                   include_self_loops=include_self_loops)

        # Forward on original graph
        out1 = layer(x, edge_index)

        # Permute graph
        x_perm, edge_index_perm, perm, inv_perm = permute_graph(x, edge_index)
        out2 = layer(x_perm, edge_index_perm)

        # For node-to-edge: edges stay in the same order, so outputs should be equal
        diff = (out1 - out2).abs().max()
        assert diff < 1e-5, f"Equivariance failed (self_loops={include_self_loops}): {diff.item()}"

    print(f"✓ GraphConvNodeToEdge equivariance test passed")


def test_edge_to_edge_equivariance():
    """Test GraphConvEdgeToEdge permutation equivariance."""
    num_nodes, num_edges = 10, 20
    edge_features, out_channels = 3, 8

    x, edge_index, edge_attr = create_random_graph(num_nodes, num_edges, 5, edge_features)

    # Test without self-loops only (with self-loops is more complex)
    layer = GraphConvEdgeToEdge(edge_features, out_channels, bias=1.0, include_self_loops=False)

    # Forward on original graph
    out1 = layer(edge_attr, edge_index, num_nodes=num_nodes)

    # Permute graph (edge_attr stays aligned with edges)
    _, edge_index_perm, perm, inv_perm = permute_graph(x, edge_index)
    out2 = layer(edge_attr, edge_index_perm, num_nodes=num_nodes)

    # For edge-to-edge, the output features stay with the edges (same order)
    # So we just check if they're the same
    diff = (out1 - out2).abs().max()
    assert diff < 1e-5, f"Equivariance failed: {diff.item()}"

    print(f"✓ GraphConvEdgeToEdge equivariance test passed (max diff: {diff.item():.2e})")


# ========================================
# Test: Batching
# ========================================

def test_node_to_node_batching():
    """Test GraphConvNodeToNode with batched graphs."""
    num_graphs = 3
    nodes_per_graph = [5, 7, 6]
    edges_per_graph = [8, 12, 10]
    in_channels, out_channels = 5, 8

    x, edge_index, batch, _ = create_batched_graph(num_graphs, nodes_per_graph, edges_per_graph, in_channels)

    layer = GraphConvNodeToNode(in_channels, out_channels, bias=1.0)
    out = layer(x, edge_index, batch=batch)

    total_nodes = sum(nodes_per_graph)
    assert out.shape == (total_nodes, out_channels), f"Expected shape ({total_nodes}, {out_channels}), got {out.shape}"

    # Verify each graph's output is independent
    # Compare batched output with individual graph outputs
    node_offset = 0
    for g in range(num_graphs):
        n_nodes = nodes_per_graph[g]
        n_edges = edges_per_graph[g]

        # Extract single graph
        x_single = x[node_offset:node_offset+n_nodes]
        edge_mask = (batch[edge_index[0]] == g)
        edge_index_single = edge_index[:, edge_mask] - node_offset

        # Forward on single graph
        out_single = layer(x_single, edge_index_single, batch=None)

        # Compare with batched output
        out_batched_slice = out[node_offset:node_offset+n_nodes]
        diff = (out_single - out_batched_slice).abs().max()
        assert diff < 1e-5, f"Batching failed for graph {g}: {diff.item()}"

        node_offset += n_nodes

    print("✓ GraphConvNodeToNode batching test passed")


def test_edge_to_node_batching():
    """Test GraphConvEdgeToNode with batched graphs."""
    num_graphs = 3
    nodes_per_graph = [5, 7, 6]
    edges_per_graph = [8, 12, 10]
    edge_features, out_channels = 3, 8

    _, edge_index, batch, edge_attr = create_batched_graph(
        num_graphs, nodes_per_graph, edges_per_graph, 5, edge_features
    )
    total_nodes = sum(nodes_per_graph)

    layer = GraphConvEdgeToNode(edge_features, out_channels, bias=1.0, include_self_loops=False)
    out = layer(edge_attr, edge_index, num_nodes=total_nodes, batch=batch)

    assert out.shape == (total_nodes, out_channels)
    print("✓ GraphConvEdgeToNode batching test passed")


# ========================================
# Test: Self-Loops
# ========================================

def test_self_loops_handling():
    """Test self-loops handling in layers that support it."""
    num_nodes = 10
    node_features, edge_features = 5, 3
    out_channels = 8

    # Create graph with explicit self-loops
    edge_index = torch.tensor([
        [0, 1, 2, 0, 1, 2],  # sources
        [1, 2, 0, 0, 1, 2],  # targets (last 3 are self-loops)
    ])
    num_edges = edge_index.size(1)

    x = torch.randn(num_nodes, node_features)
    edge_attr = torch.randn(num_edges, edge_features)

    # Test EdgeToNode with self-loops
    layer_e2n = GraphConvEdgeToNode(edge_features, out_channels, include_self_loops=True)
    out_e2n = layer_e2n(edge_attr, edge_index, num_nodes=num_nodes)
    assert out_e2n.shape == (num_nodes, out_channels)

    # Test NodeToEdge with self-loops
    layer_n2e = GraphConvNodeToEdge(node_features, out_channels, include_self_loops=True)
    out_n2e = layer_n2e(x, edge_index)
    assert out_n2e.shape == (num_edges, out_channels)

    # Test EdgeToEdge with self-loops
    layer_e2e = GraphConvEdgeToEdge(edge_features, out_channels, include_self_loops=True)
    out_e2e = layer_e2e(edge_attr, edge_index, num_nodes=num_nodes)
    assert out_e2e.shape == (num_edges, out_channels)

    print("✓ Self-loops handling test passed")


# ========================================
# Run All Tests
# ========================================

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Sparse Graph Layer Tests")
    print("="*60 + "\n")

    print("--- Basic Functionality Tests ---")
    test_node_to_node_basic()
    test_edge_to_node_basic()
    test_node_to_edge_basic()
    test_edge_to_edge_basic()

    print("\n--- Equivariance Tests ---")
    test_node_to_node_equivariance()
    test_edge_to_node_equivariance()
    test_node_to_edge_equivariance()
    test_edge_to_edge_equivariance()

    print("\n--- Batching Tests ---")
    test_node_to_node_batching()
    test_edge_to_node_batching()

    print("\n--- Self-Loops Tests ---")
    test_self_loops_handling()

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
