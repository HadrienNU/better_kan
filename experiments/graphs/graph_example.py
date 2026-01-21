"""Graph regression example using sparse graph layers on ESOL dataset.

This example demonstrates how to use GraphConvEdgeToEdge and other sparse layers
for molecular property prediction on the ESOL (water solubility) dataset.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from better_kan.graph_layers import (
    GraphConvEdgeToEdge,
    GraphConvEdgeToNode,
    GraphConvNodeToNode,
)


# ========================================
# 1. Load the ESOL Dataset
# ========================================
print("\n" + "=" * 60)
print("Loading ESOL Dataset")
print("=" * 60 + "\n")

dataset = MoleculeNet(root="/tmp/ESOL", name="ESOL")

print(f"Dataset: {dataset}")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of node features: {dataset.num_node_features}")
print(f"Number of edge features: {dataset.num_edge_features}")
print()

# Inspect a sample
sample = dataset[0]
print(f"Sample graph:")
print(f"  Nodes: {sample.num_nodes}")
print(f"  Edges: {sample.num_edges}")
print(f"  Node features shape: {sample.x.shape}")
print(f"  Edge features shape: {sample.edge_attr.shape}")
print(f"  Target (solubility): {sample.y.item():.4f}")
print()


# ========================================
# 2. Prepare DataLoaders
# ========================================
torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:900]
test_dataset = dataset[900:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training set: {len(train_dataset)} graphs")
print(f"Test set: {len(test_dataset)} graphs")
print()


# ========================================
# 3. Define the Model Using Sparse Layers
# ========================================
class EdgeGNN(torch.nn.Module):
    """GNN using edge-based convolutions with sparse graph layers.

    Architecture:
    1. Edge-to-Edge convolution (learns edge representations)
    2. Edge-to-Node aggregation (converts to node representations)
    3. Node-to-Node convolution (refines node representations)
    4. Global pooling (graph-level representation)
    5. MLP for regression
    """

    def __init__(self, node_features, edge_features, hidden_channels):
        super(EdgeGNN, self).__init__()

        # Edge processing layers
        self.edge_conv1 = GraphConvEdgeToEdge(edge_features, hidden_channels, bias=1.0)
        self.edge_conv2 = GraphConvEdgeToEdge(hidden_channels, hidden_channels, bias=1.0)

        # Edge to node aggregation
        self.edge_to_node = GraphConvEdgeToNode(hidden_channels, hidden_channels, bias=1.0)

        # Node processing
        self.node_conv = GraphConvNodeToNode(hidden_channels, hidden_channels, bias=1.0)

        # Readout MLP
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = torch.nn.Linear(hidden_channels // 2, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        num_nodes = x.size(0)

        # 1. Process edges with edge-to-edge convolutions
        edge_h = self.edge_conv1(edge_attr, edge_index, num_nodes=num_nodes, batch=batch)
        edge_h = F.relu(edge_h)
        edge_h = self.edge_conv2(edge_h, edge_index, num_nodes=num_nodes, batch=batch)
        edge_h = F.relu(edge_h)

        # 2. Aggregate edges to nodes
        node_h = self.edge_to_node(edge_h, edge_index, num_nodes=num_nodes, batch=batch)
        node_h = F.relu(node_h)

        # 3. Refine node representations
        node_h = self.node_conv(node_h, edge_index, batch=batch)
        node_h = F.relu(node_h)

        # 4. Global pooling (graph-level representation)
        graph_h = global_mean_pool(node_h, batch)

        # 5. MLP for regression
        out = F.dropout(graph_h, p=0.3, training=self.training)
        out = self.lin1(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.lin2(out)

        return out


# ========================================
# 4. Setup Device and Optimizer
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print()

model = EdgeGNN(node_features=dataset.num_node_features, edge_features=dataset.num_edge_features, hidden_channels=64).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Count parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params:,} trainable parameters")
print()


# ========================================
# 5. Training Function
# ========================================
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
        loss = criterion(out, data.y)

        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


# ========================================
# 6. Evaluation Function
# ========================================
def test(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


# ========================================
# 7. Training Loop
# ========================================
print("=" * 60)
print("Starting Training")
print("=" * 60 + "\n")

best_test_loss = float("inf")
for epoch in tqdm(range(1, 201)):
    train_loss = train()
    test_loss = test(test_loader)

    if test_loss < best_test_loss:
        best_test_loss = test_loss

    if epoch % 20 == 0:
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f} (Best: {best_test_loss:.4f})")

print(f"\nTraining completed! Best test loss: {best_test_loss:.4f}")
print()


# ========================================
# 8. Example Predictions
# ========================================
print("=" * 60)
print("Example Predictions")
print("=" * 60 + "\n")

model.eval()
with torch.no_grad():
    for i in range(5):
        sample = test_dataset[i].to(device)
        pred = model(sample.x.float(), sample.edge_index, sample.edge_attr.float(), sample.batch)
        actual = sample.y.item()
        predicted = pred.item()
        error = abs(predicted - actual)

        print(f"Molecule {i+1}:")
        print(f"  Nodes: {sample.num_nodes:2d}, Edges: {sample.num_edges:2d}")
        print(f"  Predicted: {predicted:7.4f}")
        print(f"  Actual:    {actual:7.4f}")
        print(f"  Error:     {error:7.4f}")
        print()

print("=" * 60)
print("Example completed successfully!")
print("=" * 60)
