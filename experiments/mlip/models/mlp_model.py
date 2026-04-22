"""
MLP Potential model using dense graph layers for learning atomic energies and forces.
Updated to take positions as input and compute distances internally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from better_kan.dense_graph_layers import (
    DenseGraphConvEdgeToEdge,
    DenseGraphConvEdgeToNode,
    DenseGraphConvNodeToNode,
)
from better_kan.layers import build_layer
from better_kan.functions import Splines, ChebyshevPolynomial, RBFFunction


class DistanceComputation(nn.Module):
    """Efficient distance computation for atomic positions.

    Handles both periodic and non-periodic boundary conditions.
    """

    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor | None = None,
        pbc: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute distance matrix from positions.

        Args:
            positions: Atomic positions [B, N, 3] or [N, 3]
            cell: Unit cell vectors [B, 3, 3] or [3, 3] (for periodic systems)
            pbc: Periodic boundary conditions [B, 3] or [3]
            mask: Boolean mask for padding [B, N] or [N]

        Returns:
            Distance matrix [B, N, N] or [N, N]
        """
        single_system = positions.ndim == 2

        if single_system:
            positions = positions.unsqueeze(0)
            if cell is not None:
                cell = cell.unsqueeze(0)
            if pbc is not None:
                pbc = pbc.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, N, _ = positions.shape

        # Handle periodic boundary conditions
        if cell is not None and pbc is not None and pbc.any():
            distances = self._compute_distances_mic(positions, cell, pbc)
        else:
            distances = torch.cdist(positions, positions)

        # Apply mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(1) * mask.unsqueeze(2)
            distances = distances * mask_expanded.float()

        if single_system:
            distances = distances.squeeze(0)

        return distances

    def _compute_distances_mic(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            positions: [B, N, 3]
            cell: [B, 3, 3] (cell vectors as rows)
            pbc: [B, 3] (binary/boolean mask)
        """
        B, N, _ = positions.shape

        # 1. Compute pairwise displacement vectors [B, N, N, 3]
        # Use broadcasting instead of manual expand for memory efficiency
        displacements = positions.unsqueeze(2) - positions.unsqueeze(1)

        # 2. Convert to fractional coordinates
        # Formula: frac = cart @ inv_cell
        inv_cell = torch.inverse(cell)
        frac_disp = torch.einsum("bnmi,bij->bnmj", displacements, inv_cell)

        # 3. Apply minimum image convention
        # Ensure pbc is broadcastable: [B, 1, 1, 3]
        pbc_mask = pbc.view(B, 1, 1, 3)
        # We only round the components where PBC is True
        frac_disp = frac_disp - torch.round(frac_disp) * pbc_mask

        # 4. Convert back to Cartesian coordinates
        # Formula: cart = frac @ cell
        cart_disp = torch.einsum("bnmi,bij->bnmj", frac_disp, cell)

        # 5. Compute distances
        distances = torch.norm(cart_disp, dim=-1)  # [B, N, N]

        return distances


class InverseDistanceComputation(nn.Module):
    """Efficient inverse distance computation for atomic positions.
    Handles both periodic and non-periodic boundary conditions.
    Optimized for memory and speed.
    """

    def __init__(self, cutoff: float | None = None):
        """
        Args:
            cutoff: Optional distance cutoff. Inverse distances beyond this are set to 0.
                   Helps with numerical stability and physical relevance.
        """
        super().__init__()
        self.cutoff = cutoff

    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor | None = None,
        pbc: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute inverse distance matrix from positions.

        Args:
            positions: Atomic positions [B, N, 3] or [N, 3]
            cell: Unit cell vectors [B, 3, 3] or [3, 3] (for periodic systems)
            pbc: Periodic boundary conditions [B, 3] or [3]
            mask: Boolean mask for padding [B, N] or [N]

        Returns:
            Inverse distance matrix [B, N, N] or [N, N]
            Self-interactions (diagonal) are set to 0.
        """
        single_system = positions.ndim == 2
        if single_system:
            positions = positions.unsqueeze(0)
            if cell is not None:
                cell = cell.unsqueeze(0)
            if pbc is not None:
                pbc = pbc.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, N, _ = positions.shape

        # Handle periodic boundary conditions
        if cell is not None and pbc is not None and pbc.any():
            inv_distances = self._compute_inv_distances_mic(positions, cell, pbc)
        else:
            inv_distances = self._compute_inv_distances_simple(positions)

        # Apply mask if provided
        if mask is not None:
            # More efficient masking: compute once and broadcast
            mask_expanded = mask.unsqueeze(2) & mask.unsqueeze(1)
            inv_distances.masked_fill_(~mask_expanded, 0.0)

        if single_system:
            inv_distances = inv_distances.squeeze(0)

        return inv_distances

    def _compute_inv_distances_simple(self, positions: torch.Tensor) -> torch.Tensor:
        """Optimized inverse distance computation without PBC.

        Args:
            positions: [B, N, 3]

        Returns:
            Inverse distances [B, N, N]
        """
        B, N, _ = positions.shape

        # Compute squared distances directly to avoid intermediate storage
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
        pos_sq = (positions**2).sum(dim=-1, keepdim=True)  # [B, N, 1]

        # Compute pairwise squared distances
        # Broadcasting: [B, N, 1] + [B, 1, N] - 2*[B, N, N]
        dist_sq = pos_sq + pos_sq.transpose(1, 2) - 2 * torch.bmm(positions, positions.transpose(1, 2))

        # Clamp to avoid negative values due to numerical errors
        dist_sq = dist_sq.clamp(min=0.0)

        # Create mask for non-zero distances (excludes diagonal and handles numerical zeros)
        # Diagonal elements will have dist_sq = 0
        diagonal_indices = torch.arange(N, device=positions.device)

        # Compute distances (sqrt)
        distances = torch.sqrt(dist_sq)

        # Create mask: True where we want to compute inverse (non-zero, non-diagonal)
        valid_mask = distances > 0.0
        valid_mask[:, diagonal_indices, diagonal_indices] = False

        # Apply cutoff if specified
        if self.cutoff is not None:
            valid_mask &= distances <= self.cutoff

        # Compute inverse distances only where valid
        inv_distances = torch.zeros_like(distances)
        inv_distances[valid_mask] = 1.0 / distances[valid_mask]

        return inv_distances

    def _compute_inv_distances_mic(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
    ) -> torch.Tensor:
        """Optimized inverse distance computation with minimum image convention.

        Args:
            positions: [B, N, 3]
            cell: [B, 3, 3] (cell vectors as rows)
            pbc: [B, 3] (binary/boolean mask)

        Returns:
            Inverse distances [B, N, N]
        """
        B, N, _ = positions.shape

        # Pre-compute inverse cell (reuse if called multiple times in batches)
        inv_cell = torch.linalg.inv(cell)  # More stable than torch.inverse

        # 1. Compute pairwise displacement vectors [B, N, N, 3]
        # Memory optimization: use view instead of unsqueeze where possible
        displacements = positions.view(B, N, 1, 3) - positions.view(B, 1, N, 3)

        # 2. Convert to fractional coordinates (optimized einsum)
        # Reshape for batch matrix multiplication
        disp_flat = displacements.reshape(B, N * N, 3)
        frac_disp_flat = torch.bmm(disp_flat, inv_cell.transpose(1, 2))
        frac_disp = frac_disp_flat.view(B, N, N, 3)

        # 3. Apply minimum image convention in-place
        # Only process PBC dimensions to save computation
        if pbc.all():
            # All dimensions have PBC - simpler and faster
            frac_disp.sub_(frac_disp.round())
        else:
            # Selective PBC - only round where needed
            pbc_mask = pbc.view(B, 1, 1, 3)
            frac_disp.sub_(frac_disp.round().mul_(pbc_mask))

        # 4. Convert back to Cartesian coordinates
        frac_flat = frac_disp.reshape(B, N * N, 3)
        cart_disp_flat = torch.bmm(frac_flat, cell)
        cart_disp = cart_disp_flat.view(B, N, N, 3)

        # 5. Compute distances
        distances = torch.norm(cart_disp, dim=-1)  # [B, N, N]

        # Create mask for valid inverse distances
        diagonal_indices = torch.arange(N, device=positions.device)
        valid_mask = distances > 0.0
        valid_mask[:, diagonal_indices, diagonal_indices] = False

        # Apply cutoff if specified
        if self.cutoff is not None:
            valid_mask &= distances <= self.cutoff

        # Compute inverse distances only where valid
        inv_distances = torch.zeros_like(distances)
        inv_distances[valid_mask] = 1.0 / distances[valid_mask]

        return inv_distances


class MLPPotentialXYZ(nn.Module):
    """Machine Learning Potential using dense graph layers with position input.

    Architecture:
        1. Inverse distance computation from positions
        2. Edge initialization: Combine distances + atom type pairs
        3. Edge→Edge × n_edge_layers: Process pairwise interactions
        4. Edge→Node: Aggregate to atom representations
        5. Node→Node × n_node_layers: Refine atom embeddings
        6. Global pooling: Sum/mean to get graph-level energy
        7. Force computation: Autograd from energy w.r.t. positions

    Args:
        n_atom_types: Number of atom types (elements)
        hidden_dim: Hidden dimension for layers
        n_edge_layers: Number of edge-to-edge layers
        n_node_layers: Number of node-to-node layers
        atom_embedding_dim: Dimension for atom type embeddings
        False: Whether to include self-loops in edge layers
        activation: Activation function to use
    """

    def __init__(
        self,
        n_atom_types: int,
        hidden_dim: int = 64,
        n_edge_layers: int = 2,
        n_node_layers: int = 1,
        atom_embedding_dim: int = 16,
        activation: str = "silu",
    ):
        super().__init__()

        self.n_atom_types = n_atom_types
        self.hidden_dim = hidden_dim
        self.atom_embedding_dim = atom_embedding_dim

        self.atom_embedding = nn.Embedding(n_atom_types, atom_embedding_dim)
        self.distance_computation = InverseDistanceComputation()

        edge_dim = 1 + 2 * atom_embedding_dim

        self.edge_in_proj = nn.Linear(edge_dim, hidden_dim, bias=False)

        self.edge_layers = nn.ModuleList()
        for i in range(n_edge_layers):
            self.edge_layers.append(
                DenseGraphConvEdgeToEdge(
                    hidden_dim,
                    hidden_dim,
                    bias=1.0,
                    include_self_loops=False,
                )
            )

        self.edge_to_node = DenseGraphConvEdgeToNode(
            hidden_dim,
            hidden_dim,
            bias=1.0,
            include_self_loops=False,
        )

        self.node_layers = nn.ModuleList()
        for i in range(n_node_layers):
            self.node_layers.append(
                DenseGraphConvNodeToNode(
                    hidden_dim,
                    hidden_dim,
                    bias=1.0,
                )
            )

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        if activation == "silu":
            self.act = F.silu
        elif activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def _create_edge_features(
        self,
        inv_distances: torch.Tensor,
        atom_types: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create edge features from distances and atom types.

        Args:
            inv_distances: Inverse  Distance matrices [B, N, N] or [N, N]
            atom_types: Atom type indices [B, N] or [N]
            mask: Boolean mask for padding [B, N] or [N]

        Returns:
            Edge features [B, N, N, edge_dim] or [N, N, edge_dim]
        """
        if inv_distances.ndim == 2:
            inv_distances = inv_distances.unsqueeze(0)
            atom_types = atom_types.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            single_system = True
        else:
            single_system = False

        B, N, _ = inv_distances.shape

        atom_embeddings = self.atom_embedding(atom_types)

        atom_emb_i = atom_embeddings.unsqueeze(2).expand(-1, -1, N, -1)
        atom_emb_j = atom_embeddings.unsqueeze(1).expand(-1, N, -1, -1)

        inv_distances_expanded = inv_distances.unsqueeze(-1)

        edge_features = torch.cat([inv_distances_expanded, atom_emb_i, atom_emb_j], dim=-1)

        edge_features = self.edge_in_proj(edge_features)

        if mask is not None:
            mask_edge = mask.unsqueeze(1) * mask.unsqueeze(2)
            edge_features = edge_features * mask_edge.unsqueeze(-1)

        if single_system:
            edge_features = edge_features.squeeze(0)

        return edge_features

    def forward(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_forces: bool = False,
        cell: torch.Tensor | None = None,
        pbc: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the MLP Potential.

        Args:
            positions: Atomic positions [B, N, 3] or [N, 3]
            atom_types: Atom type indices [B, N] or [N]
            mask: Boolean mask for padding [B, N] or [N]
            return_forces: Whether to compute forces via autograd
            cell: Unit cell vectors [B, 3, 3] or [3, 3] for periodic systems
            pbc: Periodic boundary conditions [B, 3] or [3]

        Returns:
            Dictionary containing:
                - energy: Predicted energies [B, 1] or [1]
                - forces: Predicted forces [B, N, 3] or [N, 3] (if return_forces)
                - node_features: Node embeddings [B, N, hidden_dim] or [N, hidden_dim]
                - edge_features: Edge embeddings [B, N, N, hidden_dim] or [N, N, hidden_dim]
        """
        if return_forces:
            if not positions.requires_grad:
                positions = positions.clone().detach().requires_grad_(True)

        single_system = positions.ndim == 2

        if single_system:
            positions = positions.unsqueeze(0)
            atom_types = atom_types.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            if cell is not None:
                cell = cell.unsqueeze(0)
            if pbc is not None:
                pbc = pbc.unsqueeze(0)

        B, N, _ = positions.shape

        # Compute distances from positions
        inv_distances = self.distance_computation(positions, cell, pbc, mask)

        edge_features = self._create_edge_features(inv_distances, atom_types, mask)

        mask_edge = None
        for layer in self.edge_layers:
            edge_features = layer(None, None, mask, edge_features)
            edge_features = self.act(edge_features)

            if mask is not None:
                mask_edge = mask.unsqueeze(1) * mask.unsqueeze(2)
                edge_features = edge_features * mask_edge.unsqueeze(-1)

        node_features = self.edge_to_node(None, None, mask, edge_features)
        node_features = self.act(node_features)

        if mask is not None:
            node_features = node_features * mask.unsqueeze(-1)

        for layer in self.node_layers:
            node_features = layer(node_features, None, mask, None)
            node_features = self.act(node_features)

            if mask is not None:
                node_features = node_features * mask.unsqueeze(-1)

        if mask is not None:
            node_features_sum = (node_features * mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
            mask_sum = mask.unsqueeze(-1).sum(dim=1, keepdim=True)
            energy = self.energy_head(node_features_sum / mask_sum.clamp(min=1.0))
        else:
            energy = self.energy_head(node_features.mean(dim=1, keepdim=True))
        result = {
            "energy": energy.squeeze(-1),
            "node_features": node_features,
            "edge_features": edge_features,
        }

        if return_forces:
            energy_sum = energy.sum()
            grad_energy = torch.autograd.grad(energy_sum, positions, create_graph=True)[0]
            forces = -grad_energy

            if mask is not None:
                forces = forces * mask.unsqueeze(-1)

            result["forces"] = forces

        if single_system:
            result["energy"] = result["energy"].squeeze(0)
            result["node_features"] = result["node_features"].squeeze(0)
            result["edge_features"] = result["edge_features"].squeeze(0)
            if "forces" in result:
                result["forces"] = result["forces"].squeeze(0)

        return result


class MLPPotentialWithKANXYZ(nn.Module):
    """MLP Potential with KAN-style layers for better approximation.

    Uses dense graph architecture but with learnable activation functions
    inspired by KAN (Kolmogorov-Arnold Networks).
    """

    def __init__(
        self,
        n_atom_types: int,
        hidden_dim: int = 64,
        n_edge_layers: int = 2,
        n_node_layers: int = 1,
        atom_embedding_dim: int = 16,
    ):
        super().__init__()

        self.n_atom_types = n_atom_types
        self.hidden_dim = hidden_dim
        self.atom_embedding_dim = atom_embedding_dim

        self.atom_embedding = nn.Embedding(n_atom_types, atom_embedding_dim)
        self.distance_computation = InverseDistanceComputation()

        edge_dim = 1 + 2 * atom_embedding_dim

        self.edge_in_proj = nn.Linear(edge_dim, hidden_dim, bias=False)

        self.edge_layers = nn.ModuleList()
        for i in range(n_edge_layers):
            layer = DenseGraphConvEdgeToEdge(
                hidden_dim,
                hidden_dim,
                bias=1.0,
                include_self_loops=False,
            )
            self.edge_layers.append(layer)

            layer = build_layer(hidden_dim, hidden_dim, ChebyshevPolynomial, poly_order=5)
            self.edge_layers.append(layer)

        self.edge_to_node = DenseGraphConvEdgeToNode(
            hidden_dim,
            hidden_dim,
            bias=1.0,
            include_self_loops=False,
        )

        self.node_layers = nn.ModuleList()
        for i in range(n_node_layers):
            layer = build_layer(hidden_dim, hidden_dim, ChebyshevPolynomial, poly_order=5)
            self.node_layers.append(layer)
            layer = DenseGraphConvNodeToNode(
                hidden_dim,
                hidden_dim,
                bias=1.0,
            )
            self.node_layers.append(layer)

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _create_edge_features(
        self,
        inv_distances: torch.Tensor,
        atom_types: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inv_distances.ndim == 2:
            inv_distances = inv_distances.unsqueeze(0)
            atom_types = atom_types.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            single_system = True
        else:
            single_system = False

        B, N, _ = inv_distances.shape

        atom_embeddings = self.atom_embedding(atom_types)

        atom_emb_i = atom_embeddings.unsqueeze(2).expand(-1, -1, N, -1)
        atom_emb_j = atom_embeddings.unsqueeze(1).expand(-1, N, -1, -1)

        inv_distances_expanded = inv_distances.unsqueeze(-1)

        edge_features = torch.cat([inv_distances_expanded, atom_emb_i, atom_emb_j], dim=-1)

        edge_features = self.edge_in_proj(edge_features)

        if mask is not None:
            mask_edge = mask.unsqueeze(1) * mask.unsqueeze(2)
            edge_features = edge_features * mask_edge.unsqueeze(-1)

        if single_system:
            edge_features = edge_features.squeeze(0)

        return edge_features

    def forward(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_forces: bool = False,
        cell: torch.Tensor | None = None,
        pbc: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if return_forces:
            if not positions.requires_grad:
                positions = positions.clone().detach().requires_grad_(True)

        single_system = positions.ndim == 2

        if single_system:
            positions = positions.unsqueeze(0)
            atom_types = atom_types.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            if cell is not None:
                cell = cell.unsqueeze(0)
            if pbc is not None:
                pbc = pbc.unsqueeze(0)

        B, N, _ = positions.shape

        # Compute inv_distances from positions
        inv_distances = self.distance_computation(positions, cell, pbc, mask)

        edge_features = self._create_edge_features(inv_distances, atom_types, mask)

        mask_edge = None
        for i, layer in enumerate(self.edge_layers):
            edge_features = layer(None, None, mask, edge_features)
            if mask is not None:
                mask_edge = mask.unsqueeze(1) * mask.unsqueeze(2)
                edge_features = edge_features * mask_edge.unsqueeze(-1)

        node_features = self.edge_to_node(None, None, mask, edge_features)

        if mask is not None:
            node_features = node_features * mask.unsqueeze(-1)

        for i, layer in enumerate(self.node_layers):
            node_features = layer(node_features, None, mask, None)
            if mask is not None:
                node_features = node_features * mask.unsqueeze(-1)

        if mask is not None:
            node_features_sum = (node_features * mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
            mask_sum = mask.unsqueeze(-1).sum(dim=1, keepdim=True)
            energy = self.energy_head(node_features_sum / mask_sum.clamp(min=1.0))
        else:
            energy = self.energy_head(node_features.mean(dim=1, keepdim=True))

        result = {
            "energy": energy.squeeze(-1),
            "node_features": node_features,
            "edge_features": edge_features,
        }

        if return_forces:
            energy_sum = energy.sum()
            grad_energy = torch.autograd.grad(energy_sum, positions, create_graph=True)[0]
            forces = -grad_energy

            if mask is not None:
                forces = forces * mask.unsqueeze(-1)

            result["forces"] = forces

        if single_system:
            result["energy"] = result["energy"].squeeze(0)
            result["node_features"] = result["node_features"].squeeze(0)
            result["edge_features"] = result["edge_features"].squeeze(0)
            if "forces" in result:
                result["forces"] = result["forces"].squeeze(0)

        return result


# Backward compatibility aliases
MLPPotential = MLPPotentialXYZ
MLPPotentialWithKAN = MLPPotentialWithKANXYZ
