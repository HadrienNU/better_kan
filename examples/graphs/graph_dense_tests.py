import torch

from better_kan.dense_graph_layers import DenseGraphConvNodeToNode, DenseGraphConvEdgeToNode, DenseGraphConvNodeToEdge, DenseGraphConvEdgeToEdge


# -------------------------------------------------------------
# Apply independent permutations per graph (with edge attributes)
# -------------------------------------------------------------
def permute_batched_graph(x, adj, edge_attr, mask, perms):
    B, N, F = x.shape
    E = edge_attr.size(-1)

    x_p = torch.zeros_like(x)
    adj_p = torch.zeros_like(adj)
    edge_attr_p = torch.zeros_like(edge_attr)
    mask_p = torch.zeros_like(mask)

    # P is stored so we can unpermute outputs later
    P_batch = torch.zeros(B, N, N)

    for b in range(B):
        perm = perms[b]
        P = torch.eye(N)[perm]
        P_batch[b] = P

        # node features
        x_p[b] = P @ x[b]

        # adjacency
        adj_p[b] = P @ adj[b] @ P.t()

        # edge attributes: E[b, i, j] → E[b, perm[i], perm[j]]
        edge_attr_p[b] = edge_attr[b][perm][:, perm]

        # mask: simply permute nodes
        mask_p[b] = mask[b][perm]

    return x_p, adj_p, edge_attr_p, mask_p, P_batch


# -------------------------------------------------------------
# Equivariance checker
# -------------------------------------------------------------
def check_equivariance_batched(layer, x, adj, mask, edge_attr):  # TODO adapt for edge equivariance
    B, N, F = x.shape

    # random per-graph permutations
    perms = [torch.randperm(N) for _ in range(B)]

    # permute everything
    x_p, adj_p, edge_attr_p, mask_p, P = permute_batched_graph(x, adj, edge_attr, mask, perms)

    # compute outputs
    out1 = layer(x, adj, mask=mask, edge_attrs=edge_attr)
    out2 = layer(x_p, adj_p, mask=mask_p, edge_attrs=edge_attr_p)

    if len(out2.shape) == 3:  # Vector case
        # unpermute output: P^T @ f(Px, P A P^T)
        out2_back = torch.matmul(P.transpose(1, 2), out2)
    else:  # matrix case unpermute output: P^T @ f(Px, P A P^T) @ P
        out2_back = torch.einsum("bnlo,blm-> bnmo", torch.einsum("bnm,bmlo-> bnlo", P.transpose(1, 2), out2), P)
    diff = (out1 - out2_back).abs().max()
    diff_noperm = (out1 - out2).abs().max()
    print(f"Max difference: {diff.item():.6f}", diff_noperm.item())
    return diff.item() < 1e-5


# -------------------------------------------------------------
# Build example batched graph with edge attributes and mask
# -------------------------------------------------------------
B = 3  # batch size
N = 5  # max nodes per graph
F_in = 4  # node features
E_in = 3  # edge attribute size

# Node features
x = torch.randn(B, N, F_in)

# Dense adjacency
adj = torch.randint(0, 2, (B, N, N)).float()
adj = torch.triu(adj, diagonal=1)
adj = adj + adj.transpose(1, 2)
for b in range(B):
    adj[b].fill_diagonal_(1.0)

# Edge attributes (dense per edge)
edge_attr = torch.randn(B, N, N, E_in)

# Node mask: mark some nodes as nonexistent
mask = torch.ones(B, N, dtype=bool)
mask[0, -2:] = 0  # graph 0 has 3 nodes
mask[1, -1] = 0  # graph 1 has 4 nodes
# graph 2 has all 5 nodes

# -------------------------------------------------------------
# Test equivariance
# -------------------------------------------------------------
print("\nTesting DenseGraphConv with edge_attr + mask:")
layer = DenseGraphConvEdgeToEdge(E_in, 6, bias=1.0)  # output channels
print("Equivariant:", check_equivariance_batched(layer, x, adj, mask, edge_attr))
