#!/usr/bin/env python3
"""
Test script to verify XYZ data loading and model functionality.
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


def test_basic_functionality():
    """Test basic data loading and model functionality."""
    print("Testing basic functionality...")

    # Test 1: Create dummy XYZ file if needed
    if not os.path.exists("test_large_forces.xyz"):
        print("Creating test XYZ file...")
        from ase import Atoms
        from ase.io import write

        # Create a simple water molecule
        atoms = Atoms(symbols=["O", "H", "H"], positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.95, 0.0, 0.0]], cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], pbc=[True, True, False])
        atoms.info["energy"] = -76.4

        write("test.xyz", atoms)
        print("Created test.xyz")

    # Test 2: Load dataset
    try:
        dataset = XYZDataset("test_large_forces.xyz")
        print(f"✓ Dataset loaded: {len(dataset)} systems")
        print(f"✓ Max atoms: {dataset.max_atoms}")
        print(f"✓ Atom types: {dataset.atom_type_mapping}")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

    # Test 3: Test data loading
    try:
        sample = dataset[0]
        print(f"✓ Sample loaded with keys: {sample.keys()}")
        print(f"✓ Positions shape: {sample['positions'].shape}")
        print(f"✓ Atom types shape: {sample['atom_types'].shape}")
        print(f"✓ Energy: {sample['energy']}")
    except Exception as e:
        print(f"✗ Sample loading failed: {e}")
        return False

    # Test 4: Test collate function
    try:
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        print(f"✓ Collate function works with batch shape: {batch['positions'].shape}")
    except Exception as e:
        print(f"✗ Collate function failed: {e}")
        return False

    # Test 5: Test model
    try:
        model = MLPPotentialXYZ(
            n_atom_types=dataset.n_atom_types,
            hidden_dim=16,  # Small for testing
            n_edge_layers=1,
            n_node_layers=1,
            atom_embedding_dim=8,
        )
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 6: Test forward pass (without force computation)
    try:
        model.eval()
        with torch.no_grad():
            # Use CPU for testing
            batch_cpu = {k: v.cpu() for k, v in batch.items()}
            output = model(
                positions=batch_cpu["positions"],
                atom_types=batch_cpu["atom_types"],
                mask=batch_cpu["mask"],
                return_forces=False,
                cell=batch_cpu["cell"],
                pbc=batch_cpu["pbc"],
            )
        print(f"✓ Forward pass successful")
        print(f"✓ Energy shape: {output['energy'].shape}")
        print(f"✓ Node features shape: {output['node_features'].shape}")
        print(f"✓ Edge features shape: {output['edge_features'].shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 7: Test force computation
    try:
        model.eval()
        batch_cpu = {k: v.cpu().requires_grad_(v.dtype == torch.float) for k, v in batch.items()}

        output = model(
            positions=batch_cpu["positions"],
            atom_types=batch_cpu["atom_types"],
            mask=batch_cpu["mask"],
            return_forces=True,
            cell=batch_cpu["cell"],
            pbc=batch_cpu["pbc"],
        )
        print(f"✓ Force computation successful")
        print(f"✓ Forces shape: {output['forces'].shape}")
        print(output["energy"], output["forces"])
    except Exception as e:
        print(f"✗ Force computation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 8: analyze distances
    print("-----------")
    distances = model.distance_computation(batch_cpu["positions"], batch_cpu["cell"], batch_cpu["pbc"], batch_cpu["mask"])
    print(distances, distances.shape)
    print(batch_cpu["atom_types"])
    print(batch_cpu["mask"])
    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n🎉 Basic functionality test passed!")
        sys.exit(0)
    else:
        print("\n❌ Basic functionality test failed!")
        sys.exit(1)
