import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM7b
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# 1. Load the ESOL dataset
# This will download the dataset to /tmp/ESOL if not present
dataset = QM7b(root="/tmp/ESOL")

print(f"Dataset: {dataset}")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of node features: {dataset.num_node_features}")
print(f"Number of edge features: {dataset.num_edge_features}")

# 2. Prepare DataLoaders
# Shuffle and split into train/test
torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:900]
test_dataset = dataset[900:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


sample = test_dataset[0]
print(sample)
print(sample.edge_attr)

# # 3. Define the Model
# class GCN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super(GCN, self).__init__()

#         # GCN Layers
#         self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, hidden_channels)

#         # Linear Layer for regression (output size = 1)
#         self.lin = torch.nn.Linear(hidden_channels, 1)

#     def forward(self, x, edge_index, batch):
#         # 1. Obtain node embeddings
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#         x = x.relu()
#         x = self.conv3(x, edge_index)

#         # 2. Readout layer (Global Pooling)
#         # Combines all node embeddings into a single graph embedding
#         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

#         # 3. Apply a final classifier
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin(x)

#         return x


# # 4. Setup Device and Optimizer
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = GCN(hidden_channels=64).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.MSELoss()


# # 5. Training Function
# def train():
#     model.train()
#     total_loss = 0
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         # Pass node features, edge connectivity, and the batch vector
#         out = model(data.x.float(), data.edge_index, data.batch)
#         # data.y contains the target solubility
#         loss = criterion(out, data.y)

#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * data.num_graphs

#     return total_loss / len(train_loader.dataset)


# # 6. Evaluation Function
# def test(loader):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             out = model(data.x.float(), data.edge_index, data.batch)
#             loss = criterion(out, data.y)
#             total_loss += loss.item() * data.num_graphs

#     return total_loss / len(loader.dataset)


# # 7. Run the Loop
# print("\nStarting training...")
# for epoch in range(1, 201):
#     train_loss = train()
#     test_loss = test(test_loader)

#     if epoch % 20 == 0:
#         print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# print("\nExample Prediction vs Actual:")
# # Let's take one molecule from the test set and see the result
# sample = test_dataset[0].to(device)
# model.eval()
# with torch.no_grad():
#     print(sample.x)
#     pred = model(sample.x.float(), sample.edge_index, sample.batch)
#     print(f"Predicted Solubility: {pred.item():.4f}")
#     print(f"Actual Solubility:    {sample.y.item():.4f}")
