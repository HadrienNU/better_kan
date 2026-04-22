"""
Working example: Using the updated ML Potential with XYZ files.

This script demonstrates the correct usage of the fixed training script
with extended XYZ files containing positions, energies, forces, and atom types.
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import XYZDataset, collate_fn
from models.mlp_model import MLPPotentialXYZ
from torch.utils.data import DataLoader


def main():
    print("🧪 Testing ML Potential with XYZ files...")

    # Create sample XYZ file if needed
    if not os.path.exists("sample_mad.xyz"):
        print("Creating sample MAD dataset...")
        from ase import Atoms
        from ase.io import write

        # Create a simple system (water molecule)
        atoms = Atoms(symbols=["O", "H", "H"], positions=[[0.0, 0.0, 0.0], [0.95, 0.0, -0.9], [0.95, 0.0, 0.9]], cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], pbc=[True, True, False])
        atoms.info["energy"] = -76.4

        # Create multiple frames
        frames = []
        for i in range(5):
            atoms_copy = atoms.copy()
            atoms_copy.positions = atoms.positions + np.random.randn(3, 3) * 0.1
            atoms_copy.info["energy"] = -76.4 + np.random.randn() * 2.0
            frames.append(atoms_copy)

        write("sample_mad.xyz", frames)
        print("✓ Created sample_mad.xyz with 5 frames")

    # Load dataset
    try:
        dataset = XYZDataset("sample_mad.xyz")
        print(f"✓ Dataset loaded: {len(dataset)} systems")
        print(f"✓ Atom types: {dataset.atom_type_mapping}")
        print(f"✓ Max atoms: {dataset.max_atoms}")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return

    # Create model
    try:
        model = MLPPotentialXYZ(
            n_atom_types=dataset.n_atom_types,
            hidden_dim=32,  # Small for testing
            n_edge_layers=1,
            n_node_layers=1,
            atom_embedding_dim=8,
        )
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return

    # Test data loading
    try:
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        print(f"✓ Batch created with shapes:")
        for key, value in batch.items():
            print(f"   {key}: {value.shape}")
    except Exception as e:
        print(f"✗ Batch creation failed: {e}")
        return

    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            batch_cpu = {k: v.cpu().requires_grad_(v.dtype == torch.float) for k, v in batch.items()}

            # Test without forces
            output_no_forces = model(
                positions=batch_cpu["positions"],
                atom_types=batch_cpu["atom_types"],
                mask=batch_cpu["mask"],
                cell=batch_cpu["cell"],
                pbc=batch_cpu["pbc"],
                return_forces=False,
            )
            print(f"✓ Energy prediction shape: {output_no_forces['energy'].shape}")

        # Test with forces
        # batch_cpu["positions"] = batch_cpu["positions"].requires_grad_(True)
        batch_cpu = {k: v.cpu().requires_grad_(v.dtype == torch.float) for k, v in batch.items()}
        output_with_forces = model(
            positions=batch_cpu["positions"],
            atom_types=batch_cpu["atom_types"],
            mask=batch_cpu["mask"],
            cell=batch_cpu["cell"],
            pbc=batch_cpu["pbc"],
            return_forces=True,
        )

        print(f"✓ Force prediction shape: {output_with_forces['forces'].shape}")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n🎉 All tests passed! The ML Potential is working correctly.")
    print("\n📋 To train on your data:")
    print("   python train.py --train_file mad-train.xyz --val_file mad-val.xyz --test_file mad-test.xyz")
    print("   --use_kan (optional)  # For KAN-style model")
    print("   --lambda_force 0.1  # For force training")


if __name__ == "__main__":
    main()
