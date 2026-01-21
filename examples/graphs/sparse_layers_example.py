"""Example usage of sparse graph convolution layers with PyTorch Geometric format.

This example demonstrates:
1. Basic usage of all four layer types
2. Batching multiple graphs together
3. Using self-loops
4. Integration with PyTorch Geometric datasets
"""

import torch
from better_kan.graph_layers import (
    GraphConvNodeToNode,
    GraphConvEdgeToNode,
    GraphConvNodeToEdge,
    GraphConvEdgeToEdge,
)


def example_basic_usage():
    """Basic usage example for each layer type."""
    print("\n" + "="*60)
    print("Basic Usage Examples")
    print("="*60 + "\n")

    # Create a simple graph: triangle with 3 nodes
    num_nodes = 3
    edge_index = torch.tensor([
        [0, 1, 2, 0],  # sources
        [1, 2, 0, 0],  # targets (last edge is self-loop)
    ])
    num_edges = edge_index.size(1)

    # Node features
    x = torch.randn(num_nodes, 5)
    print(f"Node features shape: {x.shape}")

    # Edge features
    edge_attr = torch.randn(num_edges, 3)
    print(f"Edge features shape: {edge_attr.shape}")
    print(f"Edge index:\n{edge_index.t()}")
    print()

    # 1. Node → Node transformation
    print("1. GraphConvNodeToNode")
    layer_n2n = GraphConvNodeToNode(5, 8)
    out_n2n = layer_n2n(x, edge_index)
    print(f"   Input: {x.shape} → Output: {out_n2n.shape}")
    print()

    # 2. Edge → Node transformation
    print("2. GraphConvEdgeToNode")
    layer_e2n = GraphConvEdgeToNode(3, 8)
    out_e2n = layer_e2n(edge_attr, edge_index, num_nodes=num_nodes)
    print(f"   Input: {edge_attr.shape} → Output: {out_e2n.shape}")
    print()

    # 3. Node → Edge transformation
    print("3. GraphConvNodeToEdge")
    layer_n2e = GraphConvNodeToEdge(5, 8)
    out_n2e = layer_n2e(x, edge_index)
    print(f"   Input: {x.shape} → Output: {out_n2e.shape}")
    print()

    # 4. Edge → Edge transformation
    print("4. GraphConvEdgeToEdge")
    layer_e2e = GraphConvEdgeToEdge(3, 8)
    out_e2e = layer_e2e(edge_attr, edge_index, num_nodes=num_nodes)
    print(f"   Input: {edge_attr.shape} → Output: {out_e2e.shape}")
    print()


def example_batching():
    """Example with batched graphs using PyG batch vector."""
    print("\n" + "="*60)
    print("Batching Example")
    print("="*60 + "\n")

    # Create 3 graphs with different sizes
    # Graph 0: 4 nodes, 5 edges
    # Graph 1: 3 nodes, 4 edges
    # Graph 2: 5 nodes, 6 edges

    x_0 = torch.randn(4, 5)
    x_1 = torch.randn(3, 5)
    x_2 = torch.randn(5, 5)
    x = torch.cat([x_0, x_1, x_2], dim=0)  # [12, 5]

    # Edge indices (with offsets for each graph)
    edge_index_0 = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 0]])
    edge_index_1 = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 0]]) + 4  # offset by 4
    edge_index_2 = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 0]]) + 7  # offset by 7
    edge_index = torch.cat([edge_index_0, edge_index_1, edge_index_2], dim=1)  # [2, 15]

    # Batch vector: indicates which graph each node belongs to
    batch = torch.tensor([0, 0, 0, 0,  # graph 0
                         1, 1, 1,      # graph 1
                         2, 2, 2, 2, 2])  # graph 2

    print(f"Total nodes: {x.shape[0]} (from 3 graphs with 4, 3, 5 nodes)")
    print(f"Total edges: {edge_index.shape[1]}")
    print(f"Batch vector: {batch.tolist()}")
    print()

    # Forward pass with batching
    layer = GraphConvNodeToNode(5, 8)
    out = layer(x, edge_index, batch=batch)
    print(f"Batched output shape: {out.shape}")
    print("Global aggregation is computed per-graph, not across all graphs")
    print()


def example_self_loops():
    """Example with self-loops."""
    print("\n" + "="*60)
    print("Self-Loops Example")
    print("="*60 + "\n")

    num_nodes = 4
    # Graph with explicit self-loops on nodes 0, 1, 2
    edge_index = torch.tensor([
        [0, 1, 2, 3, 0, 1, 2],  # sources
        [1, 2, 3, 0, 0, 1, 2],  # targets (last 3 are self-loops)
    ])
    edge_attr = torch.randn(edge_index.size(1), 3)

    print(f"Edge index (last 3 are self-loops):\n{edge_index.t()}")
    print()

    # Compare layers with and without self-loop handling
    print("EdgeToNode layer:")
    layer_no_sl = GraphConvEdgeToNode(3, 8, include_self_loops=False)
    layer_with_sl = GraphConvEdgeToNode(3, 8, include_self_loops=True)

    out_no_sl = layer_no_sl(edge_attr, edge_index, num_nodes=num_nodes)
    out_with_sl = layer_with_sl(edge_attr, edge_index, num_nodes=num_nodes)

    print(f"  Without self-loop handling: {layer_no_sl.linears.__len__()} basis functions")
    print(f"  With self-loop handling: {layer_with_sl.linears.__len__()} basis functions")
    print(f"  Output shape: {out_no_sl.shape}")
    print()


def example_network():
    """Example building a simple GNN with multiple layers."""
    print("\n" + "="*60)
    print("Multi-Layer Network Example")
    print("="*60 + "\n")

    import torch.nn as nn

    class SimpleGNN(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            # Node → Node → Node
            self.conv1 = GraphConvNodeToNode(in_channels, hidden_channels)
            self.conv2 = GraphConvNodeToNode(hidden_channels, out_channels)
            self.relu = nn.ReLU()

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = self.relu(x)
            x = self.conv2(x, edge_index)
            return x

    # Create network
    model = SimpleGNN(5, 16, 2)

    # Create graph
    x = torch.randn(6, 5)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])

    # Forward pass
    out = model(x, edge_index)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Network has {sum(p.numel() for p in model.parameters())} parameters")
    print()


if __name__ == "__main__":
    example_basic_usage()
    example_batching()
    example_self_loops()
    example_network()

    print("="*60)
    print("Examples completed successfully!")
    print("="*60)
