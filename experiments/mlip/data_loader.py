"""
Data loading and preprocessing for atomic systems from extended XYZ files.

Supports variable-sized systems with proper masking for dense graph operations.
Uses ASE to read extended XYZ files and computes distances on-demand.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from ase.io import read


class XYZDataset(Dataset):
    """Dataset for atomic systems from extended XYZ files.

    Args:
        xyz_file: Path to extended XYZ file
        max_atoms: Maximum number of atoms for padding. If None, inferred from data.
        atom_type_mapping: Dict mapping element symbols to indices. If None, inferred.
        use_pbc: Whether to use periodic boundary conditions for distance computation
    """

    def __init__(
        self,
        xyz_file: str,
        max_atoms: int | None = None,
        atom_type_mapping: dict[str, int] | None = None,
        use_pbc: bool = True,
    ):
        self.xyz_file = xyz_file
        self.use_pbc = use_pbc

        # Read all frames to get metadata
        try:
            self.trajectory = read(xyz_file, index=":")
        except Exception as e:
            print(f"Error reading file {xyz_file}: {e}")
            raise

        self.n_systems = len(self.trajectory)

        # Extract data from trajectory
        self.positions = []
        self.atom_symbols = []
        self.energies = []
        self.forces = []
        self.cells = []
        self.pbc_flags = []

        for atoms in self.trajectory:
            # Get atomic symbols
            symbols = atoms.get_chemical_symbols()
            self.atom_symbols.append(symbols)

            # Get positions
            positions = atoms.get_positions()
            self.positions.append(positions)

            # Get energy
            try:
                energy = atoms.get_potential_energy()
                self.energies.append(float(energy))
            except:
                # Try to get from info if no calculator
                energy = atoms.info.get("energy", 0.0)
                self.energies.append(float(energy))

            # Get forces
            try:
                forces = atoms.get_forces()
                self.forces.append(forces)
            except:
                # Try to get from arrays if no calculator
                forces = atoms.arrays.get("forces", np.zeros((len(symbols), 3)))
                self.forces.append(forces)

            # Get cell and PBC info
            cell = atoms.get_cell(complete=True)
            self.cells.append(cell.array)

            pbc = atoms.get_pbc()
            self.pbc_flags.append(pbc)

        # self.positions = np.array(self.positions)
        # self.energies = np.array(self.energies)
        # self.forces = np.array(self.forces)
        # self.cells = np.array(self.cells)
        # self.pbc_flags = np.array(self.pbc_flags)

        self.n_atoms_per_system = np.array([len(symbols) for symbols in self.atom_symbols])

        if max_atoms is None:
            self.max_atoms = int(self.n_atoms_per_system.max())
        else:
            self.max_atoms = max_atoms

        # Create atom type mapping from symbols
        all_symbols = np.unique(np.concatenate(self.atom_symbols))
        if atom_type_mapping is None:
            self.atom_type_mapping = {symbol: i for i, symbol in enumerate(sorted(all_symbols))}
        else:
            self.atom_type_mapping = atom_type_mapping

        self.n_atom_types = len(self.atom_type_mapping)
        self.reverse_mapping = {v: k for k, v in self.atom_type_mapping.items()}

        # Convert symbols to indices
        self.atom_types_indices = []
        for symbols in self.atom_symbols:
            indices = np.array([self.atom_type_mapping[symbol] for symbol in symbols])
            self.atom_types_indices.append(indices)

    def __len__(self) -> int:
        return self.n_systems

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        positions = self.positions[idx]
        atom_types_indices = self.atom_types_indices[idx]
        energy = float(self.energies[idx])
        forces = self.forces[idx]
        cell = self.cells[idx]
        pbc = self.pbc_flags[idx]

        n_atoms = positions.shape[0]

        mask = torch.ones(n_atoms, dtype=torch.bool)

        positions_tensor = torch.tensor(positions, dtype=torch.float32)
        atom_types_tensor = torch.tensor(atom_types_indices, dtype=torch.long)
        energy_tensor = torch.tensor(energy, dtype=torch.float32)
        forces_tensor = torch.tensor(forces, dtype=torch.float32)
        cell_tensor = torch.tensor(cell, dtype=torch.float32)
        pbc_tensor = torch.tensor(pbc, dtype=torch.bool)

        result = {
            "positions": positions_tensor,
            "atom_types": atom_types_tensor,
            "mask": mask,
            "energy": energy_tensor,
            "forces": forces_tensor,
            "cell": cell_tensor,
            "pbc": pbc_tensor,
            "n_atoms": n_atoms,
        }

        return result


def create_atom_type_embedding(atom_types: torch.Tensor, n_atom_types: int, embedding_dim: int) -> torch.Tensor:
    """Create one-hot encoding for atom types.

    Args:
        atom_types: Tensor of atom type indices [N] or [B, N]
        n_atom_types: Total number of atom types
        embedding_dim: Embedding dimension

    Returns:
        One-hot embeddings [N, embedding_dim] or [B, N, embedding_dim]
    """
    if atom_types.ndim == 1:
        one_hot = torch.zeros(atom_types.shape[0], n_atom_types, dtype=torch.float32)
        one_hot.scatter_(1, atom_types.unsqueeze(1), 1.0)
        return one_hot
    else:
        one_hot = torch.zeros(atom_types.shape[0], atom_types.shape[1], n_atom_types, dtype=torch.float32)
        one_hot.scatter_(2, atom_types.unsqueeze(2), 1.0)
        return one_hot


def pad_to_max_size(arr: np.ndarray, max_size: int, fill_value: float = 0.0) -> np.ndarray:
    """Pad array to max_size.

    Args:
        arr: Input array (1D, 2D, or 3D)
        max_size: Target size for first dimension
        fill_value: Value to use for padding

    Returns:
        Padded array
    """
    if arr.ndim == 1:
        n = len(arr)
        padded = np.full(max_size, fill_value, dtype=arr.dtype)
        padded[:n] = arr
    elif arr.ndim == 2:
        n_rows, n_cols = arr.shape
        padded = np.full((max_size, n_cols), fill_value, dtype=arr.dtype)
        padded[:n_rows, :] = arr
    elif arr.ndim == 3:
        n_rows, n_cols, n_feats = arr.shape
        padded = np.full((max_size, n_cols, n_feats), fill_value, dtype=arr.dtype)
        padded[:n_rows, :, :] = arr
    else:
        raise ValueError(f"Unsupported array dimension: {arr.ndim}")

    return padded


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader with variable-sized systems.

    Args:
        batch: List of dictionaries from XYZDataset

    Returns:
        Batched dictionary with padded tensors and masks
    """
    max_atoms = max(item["positions"].shape[0] for item in batch)
    batch_size = len(batch)
    positions_batch = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    atom_types_batch = torch.zeros(batch_size, max_atoms, dtype=torch.long)
    mask_batch = torch.zeros(batch_size, max_atoms, dtype=torch.bool)
    energy_batch = torch.zeros(batch_size, 1, dtype=torch.float32)
    forces_batch = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    cell_batch = torch.zeros(batch_size, 3, 3, dtype=torch.float32)
    pbc_batch = torch.zeros(batch_size, 3, dtype=torch.bool)
    n_atoms_batch = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["positions"].shape[0]
        positions_batch[i, :n, :] = item["positions"]
        atom_types_batch[i, :n] = item["atom_types"]
        mask_batch[i, :n] = True
        energy_batch[i] = item["energy"]
        forces_batch[i, :n, :] = item["forces"]
        cell_batch[i, :, :] = item["cell"]
        pbc_batch[i, :] = item["pbc"]
        n_atoms_batch[i] = n
    result = {
        "positions": positions_batch,
        "atom_types": atom_types_batch,
        "mask": mask_batch,
        "energy": energy_batch,
        "forces": forces_batch,
        "cell": cell_batch,
        "pbc": pbc_batch,
        "n_atoms": n_atoms_batch,
    }

    return result


def explore_dataset(dataset_path: str) -> None:
    """Print dataset statistics and basic information.

    Args:
        dataset_path: Path to XYZ dataset file
    """
    print("=" * 60)
    print("Dataset Exploration")
    print("=" * 60)

    dataset = XYZDataset(dataset_path)

    print(f"\nDataset size: {len(dataset)} systems")
    print(f"Max atoms per system: {dataset.max_atoms}")
    print(f"Atom types: {dataset.atom_type_mapping}")
    print(f"Number of atom types: {dataset.n_atom_types}")

    print(f"\nAtoms per system:")
    print(f"  Min: {dataset.n_atoms_per_system.min()}")
    print(f"  Max: {dataset.n_atoms_per_system.max()}")
    print(f"  Mean: {dataset.n_atoms_per_system.mean():.2f}")
    print(f"  Median: {np.median(dataset.n_atoms_per_system):.2f}")

    all_distances = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        n = sample["n_atoms"]
        pos = sample["positions"][:n, :]
        pbc = sample["pbc"]
        cell = sample["cell"]

        # Compute distances using torch
        pos_tensor = pos.clone().detach()  # torch.tensor(pos)
        if pbc.any():
            # Simple Euclidean for exploration (ignoring periodic boundaries)
            dist = torch.cdist(pos_tensor, pos_tensor)
        else:
            dist = torch.cdist(pos_tensor, pos_tensor)

        upper_tri = torch.triu(dist, diagonal=1)
        all_distances.append(upper_tri[upper_tri > 0].numpy())

    all_distances = np.concatenate(all_distances)

    print(f"\nDistance statistics (sampled):")
    print(f"  Min: {all_distances.min():.4f}")
    print(f"  Max: {all_distances.max():.4f}")
    print(f"  Mean: {all_distances.mean():.4f}")
    print(f"  Std: {all_distances.std():.4f}")

    print(f"\nEnergy statistics:")
    print(f"  Min: {np.array(dataset.energies).min():.6f}")
    print(f"  Max: {np.array(dataset.energies).max():.6f}")
    print(f"  Mean: {np.array(dataset.energies).mean():.6f}")
    print(f"  Std: {np.array(dataset.energies).std():.6f}")

    all_forces = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        n = sample["n_atoms"]
        forces = sample["forces"][:n, :]
        all_forces.append(forces.numpy())

    all_forces = np.concatenate(all_forces)

    print(f"\nForce statistics (sampled):")
    print(f"  Min: {all_forces.min():.6f}")
    print(f"  Max: {all_forces.max():.6f}")
    print(f"  Mean: {all_forces.mean():.6f}")
    print(f"  Std: {all_forces.std():.6f}")

    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  Positions: {sample['positions'].shape}")
    print(f"  Atom types: {sample['atom_types'].shape}")
    print(f"  Mask: {sample['mask'].shape}")
    print(f"  Energy: {sample['energy'].shape}")
    print(f"  Forces: {sample['forces'].shape}")
    print(f"  Cell: {sample['cell'].shape}")
    print(f"  PBC: {sample['pbc'].shape}")

    print("=" * 60)
