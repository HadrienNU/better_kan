import torch

from better_kan.graph_layers import GraphConvNodeNode

N = 5
d_in = 1
d_out = 1

edge_index = torch.tensor([[0, 1, 2, 2], [1, 2, 0, 1]])
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # node 0  # node 1  # node 2

layer = GraphConvNodeNode(X.shape[1], 2)
Y = layer(X, edge_index)

print(Y.shape)  # (5, 6)

print("\Output:\n", Y)
perm = torch.randperm(X.size(0))
X_perm = X[perm]

# Permute edge_index accordingly
edge_index_perm = edge_index.clone()
# Map original node indices to permuted indices
mapping = torch.zeros_like(perm)
mapping[perm] = torch.arange(X.size(0))
edge_index_perm[0] = mapping[edge_index[0]]  # sources
edge_index_perm[1] = mapping[edge_index[1]]  # targets

Y_perm = layer(X_perm, edge_index_perm)
print("\nPermuted output:\n", Y_perm)


Y_orig_perm = Y[perm]
print("\nPermuted original output (should match):\n", Y_orig_perm)

# Difference
diff = (Y_perm - Y_orig_perm).abs().max()
print("\nMax difference after permutation:", diff.item())
