"""
Training script for MLP Potential model.

Supports both energy-only and energy+force training with proper masking.
"""

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import XYZDataset, collate_fn, explore_dataset
from models.mlp_model import MLPPotentialXYZ, MLPPotentialWithKANXYZ
import yaml

import better_kan


def parse_args():
    # Setup parser with a --config argument
    parser = argparse.ArgumentParser(description="Train MLP Potential")

    # First pass: look only for the --config argument
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args, remaining_argv = parser.parse_known_args()

    parser.add_argument("--train_file", type=str, help="Path to training XYZ file")
    parser.add_argument("--val_file", type=str, help="Path to validation XYZ file")
    parser.add_argument("--test_file", type=str, help="Path to test XYZ file")
    parser.add_argument("--dataset", type=str, help="Path to single XYZ file (for backward compatibility)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")

    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--n_edge_layers", type=int, default=2, help="Number of edge layers")
    parser.add_argument("--n_node_layers", type=int, default=1, help="Number of node layers")
    parser.add_argument("--atom_embedding_dim", type=int, default=16, help="Atom embedding dimension")
    parser.add_argument("--use_kan", action="store_true", help="Use KAN-style model")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "lbfgs"], help="Optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")

    parser.add_argument("--lambda_energy", type=float, default=1.0, help="Energy loss weight")
    parser.add_argument("--lambda_force", type=float, default=0.1, help="Force loss weight")

    parser.add_argument("--train_split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--explore", action="store_true", help="Explore dataset and exit")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Load YAML from the path found in the first pass
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            parser.set_defaults(**yaml.safe_load(f))

    # Second pass: parse everything else
    return parser.parse_args(remaining_argv, namespace=args)


def compute_loss(
    model_output: dict,
    batch: dict,
    lambda_energy: float,
    lambda_force: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Compute combined energy and force loss.

    Args:
        model_output: Output from model
        batch: Batch dictionary
        lambda_energy: Energy loss weight
        lambda_force: Force loss weight
        device: Device for computation

    Returns:
        Total loss and dictionary of individual losses
    """
    losses = {}

    energy_pred = model_output["energy"]
    energy_true = batch["energy"]

    energy_loss = F.mse_loss(energy_pred, energy_true)
    losses["energy"] = energy_loss.item()

    total_loss = lambda_energy * energy_loss

    if "forces" in model_output and "forces" in batch:
        forces_pred = model_output["forces"]
        forces_true = batch["forces"]

        mask = batch["mask"]
        mask_expanded = mask.unsqueeze(-1).expand_as(forces_true)

        forces_true_masked = forces_true * mask_expanded
        forces_pred_masked = forces_pred * mask_expanded

        force_loss = F.mse_loss(forces_pred_masked, forces_true_masked)
        losses["force"] = force_loss.item()

        total_loss += lambda_force * force_loss

    return total_loss, losses


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_losses,
    val_losses,
    path: str,
):
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        train_losses: Training loss history
        val_losses: Validation loss history
        path: Path to save checkpoint
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        path,
    )


def train_epoch(
    model,
    dataloader,
    optimizer,
    lambda_energy: float,
    lambda_force: float,
    device: torch.device,
    use_lbfgs: bool,
    batch_size: int,
):
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        lambda_energy: Energy loss weight
        lambda_force: Force loss weight
        device: Device for computation
        use_lbfgs: Whether using LBFGS optimizer
        batch_size: Batch size for LBFGS

    Returns:
        Dictionary of average losses
    """
    model.train()

    if use_lbfgs:
        closure_batch_losses = {"energy": [], "force": []}

        def closure():
            total_loss = 0.0

            for batch in dataloader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                positions = batch["positions"]
                optimizer.zero_grad()

                model_output = model(
                    positions,
                    batch["atom_types"],
                    batch["mask"],
                    return_forces=(lambda_force > 0),
                    cell=batch["cell"],
                    pbc=batch["pbc"],
                )

                batch_loss, loss_dict = compute_loss(model_output, batch, lambda_energy, lambda_force, device)
                total_loss += batch_loss

                closure_batch_losses["energy"].append(loss_dict["energy"])
                if "force" in loss_dict:
                    closure_batch_losses["force"].append(loss_dict["force"])

            return total_loss

        loss = optimizer.step(closure)

        return {
            "energy": np.mean(closure_batch_losses["energy"]),
            "force": np.mean(closure_batch_losses["force"]) if closure_batch_losses["force"] else None,
            "total": loss.item(),
        }

    else:
        epoch_losses = {"energy": [], "force": [], "total": []}

        for batch in tqdm(dataloader, desc="Training", leave=False):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            optimizer.zero_grad()

            model_output = model(
                batch["positions"],
                batch["atom_types"],
                batch["mask"],
                return_forces=(lambda_force > 0),
                cell=batch["cell"],
                pbc=batch["pbc"],
            )

            loss, loss_dict = compute_loss(model_output, batch, lambda_energy, lambda_force, device)

            loss.backward()
            optimizer.step()

            epoch_losses["energy"].append(loss_dict["energy"])
            if "force" in loss_dict:
                epoch_losses["force"].append(loss_dict["force"])
            epoch_losses["total"].append(loss.item())

        return {
            "energy": np.mean(epoch_losses["energy"]),
            "force": np.mean(epoch_losses["force"]) if epoch_losses["force"] else None,
            "total": np.mean(epoch_losses["total"]),
        }


def evaluate(
    model,
    dataloader,
    lambda_energy: float,
    lambda_force: float,
    device: torch.device,
):
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        lambda_energy: Energy loss weight
        lambda_force: Force loss weight
        device: Device for computation

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    epoch_losses = {"energy": [], "force": [], "total": []}

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            model_output = model(
                batch["positions"],
                batch["atom_types"],
                batch["mask"],
                return_forces=False,
                cell=batch["cell"],
                pbc=batch["pbc"],
            )

            loss, loss_dict = compute_loss(model_output, batch, lambda_energy, lambda_force, device)

            epoch_losses["energy"].append(loss_dict["energy"])
            if "force" in loss_dict:
                epoch_losses["force"].append(loss_dict["force"])
            epoch_losses["total"].append(loss.item())

    return {
        "energy": np.mean(epoch_losses["energy"]),
        "force": np.mean(epoch_losses["force"]) if epoch_losses["force"] else None,
        "total": np.mean(epoch_losses["total"]),
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.explore:
        if args.dataset:
            explore_dataset(args.dataset + "_train.xyz")
        elif args.train_file:
            explore_dataset(args.train_file)
        else:
            print("Error: No dataset file provided for exploration")
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Determine dataset loading strategy
    if args.train_file and args.val_file and args.test_file:
        # Separate files for each split
        print(f"Loading datasets from separate files:")
        print(f"  Train: {args.train_file}")
        print(f"  Val: {args.val_file}")
        print(f"  Test: {args.test_file}")

        train_dataset = XYZDataset(args.train_file)
        val_dataset = XYZDataset(args.val_file, max_atoms=train_dataset.max_atoms, atom_type_mapping=train_dataset.atom_type_mapping)
        test_dataset = XYZDataset(args.test_file, max_atoms=train_dataset.max_atoms, atom_type_mapping=train_dataset.atom_type_mapping)
    elif args.dataset:
        # Backward compatibility: single dataset file
        print(f"Loading dataset from: {args.dataset}")
        train_dataset = XYZDataset(args.dataset + "_train.xyz")
        # For backward compatibility, use same file for all splits (you might want to split it)
        val_dataset = XYZDataset(args.dataset + "_val.xyz", max_atoms=train_dataset.max_atoms, atom_type_mapping=train_dataset.atom_type_mapping)
        test_dataset = XYZDataset(args.dataset + "_test.xyz", max_atoms=train_dataset.max_atoms, atom_type_mapping=train_dataset.atom_type_mapping)
    else:
        print("Error: Must provide either --dataset or --train_file/--val_file/--test_file")
        return

    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Max atoms: {train_dataset.max_atoms}")
    print(f"Atom types: {train_dataset.atom_type_mapping}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    if args.use_kan:
        model = MLPPotentialWithKANXYZ(
            n_atom_types=train_dataset.n_atom_types,
            hidden_dim=args.hidden_dim,
            n_edge_layers=args.n_edge_layers,
            n_node_layers=args.n_node_layers,
            atom_embedding_dim=args.atom_embedding_dim,
        )
    else:
        model = MLPPotentialXYZ(
            n_atom_types=train_dataset.n_atom_types,
            hidden_dim=args.hidden_dim,
            n_edge_layers=args.n_edge_layers,
            n_node_layers=args.n_node_layers,
            atom_embedding_dim=args.atom_embedding_dim,
        )

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    use_lbfgs = args.optimizer.lower() == "lbfgs"

    if use_lbfgs:
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=args.learning_rate,
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe",
        )
        print("Using LBFGS optimizer")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        print("Using Adam optimizer")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        print(f"Resumed from checkpoint: {args.resume}")
    else:
        start_epoch = 0
        train_losses = []
        val_losses = []

    best_val_loss = float("inf")

    print("\nStarting training...")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            lambda_energy=args.lambda_energy,
            lambda_force=args.lambda_force,
            device=device,
            use_lbfgs=use_lbfgs,
            batch_size=args.batch_size,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            lambda_energy=args.lambda_energy,
            lambda_force=args.lambda_force,
            device=device,
        )

        train_losses.append(train_metrics)
        val_losses.append(val_metrics)

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            save_checkpoint(model, optimizer, epoch, train_losses, val_losses, best_checkpoint_path)
            print(f"  Saved best model: {best_checkpoint_path}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, train_losses, val_losses, checkpoint_path)

        force_str = f", Force: {train_metrics['force']:.6f}" if train_metrics["force"] else ""
        print(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train: Energy={train_metrics['energy']:.6f}{force_str}, Total={train_metrics['total']:.6f} | "
            f"Val: Energy={val_metrics['energy']:.6f}, Total={val_metrics['total']:.6f}"
        )

    print("\nTraining completed!")

    test_metrics = evaluate(
        model,
        test_loader,
        lambda_energy=args.lambda_energy,
        lambda_force=args.lambda_force,
        device=device,
    )

    print(f"\nTest metrics:")
    print(f"  Energy: {test_metrics['energy']:.6f}")
    if test_metrics["force"]:
        print(f"  Force: {test_metrics['force']:.6f}")
    print(f"  Total: {test_metrics['total']:.6f}")

    final_checkpoint_path = os.path.join(args.output_dir, "final_model.pt")
    save_checkpoint(model, optimizer, args.epochs - 1, train_losses, val_losses, final_checkpoint_path)

    np.savez(
        os.path.join(args.output_dir, "training_history.npz"),
        train_losses=np.array([l["total"] for l in train_losses]),
        val_losses=np.array([l["total"] for l in val_losses]),
        train_energy=np.array([l["energy"] for l in train_losses]),
        val_energy=np.array([l["energy"] for l in val_losses]),
    )

    print(f"Saved final model and training history to {args.output_dir}")


if __name__ == "__main__":
    main()
