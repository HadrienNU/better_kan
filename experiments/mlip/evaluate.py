"""
Evaluation script for trained MLP Potential model.

Computes various metrics and generates analysis plots.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_loader import XYZDataset, collate_fn
from models.mlp_model import MLPPotentialXYZ, MLPPotentialWithKANXYZ
import yaml
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MLP Potential")

    parser.add_argument("--dataset", type=str, required=True, help="Path to npz dataset")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Output directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml")

    parser.add_argument("--use_kan", action="store_true", help="Use KAN-style model")

    parser.add_argument("--lambda_energy", type=float, default=1.0, help="Energy loss weight")
    parser.add_argument("--lambda_force", type=float, default=0.1, help="Force loss weight")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")

    return parser.parse_args()


def evaluate_model(
    model,
    dataloader,
    dataset,
    lambda_energy: float,
    lambda_force: float,
    device: torch.device,
):
    """Evaluate model and compute detailed metrics.

    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        dataset: Dataset instance
        lambda_energy: Energy loss weight
        lambda_force: Force loss weight
        device: Device for computation

    Returns:
        Dictionary of evaluation metrics and predictions
    """
    model.eval()

    energy_preds = []
    energy_trues = []

    force_preds = []
    force_trues = []

    for batch in tqdm(dataloader, desc="Evaluation"):
        batch = {k: v.to(device).requires_grad_(v.dtype == torch.float) if torch.is_tensor(v) else v for k, v in batch.items()}

        model_output = model(
            positions=batch["positions"],
            atom_types=batch["atom_types"],
            mask=batch["mask"],
            return_forces=lambda_force > 0,
            cell=batch["cell"],
            pbc=batch["pbc"],
        )
        energy_pred = model_output["energy"].detach().cpu().numpy()
        energy_true = batch["energy"].detach().cpu().numpy()

        # Print any value

        if np.any(np.abs(energy_pred - energy_true) > 1e12):
            print(batch["atom_types"].cpu().numpy())  # [np.abs(energy_pred - energy_true).ravel() > 1e12])
            print(energy_pred.ravel())  # [np.abs(energy_pred - energy_true).ravel() > 1e12])
            print(energy_true.ravel())  # [np.abs(energy_pred - energy_true).ravel() > 1e12])

        energy_preds.append(energy_pred)
        energy_trues.append(energy_true)

        if "forces" in model_output and "forces" in batch:
            force_pred = model_output["forces"].detach().cpu().numpy()
            force_true = batch["forces"].detach().cpu().numpy()

            B, n_atoms, dim = force_true.shape
            forces_pred_padded = np.zeros((B, dataset.max_atoms, dim), dtype=force_pred.dtype)
            forces_pred_padded[:, :n_atoms, :] = force_pred

            forces_true_padded = np.zeros((B, dataset.max_atoms, dim), dtype=force_true.dtype)
            forces_true_padded[:, :n_atoms, :] = force_true

            force_preds.append(forces_pred_padded)
            force_trues.append(forces_true_padded)
    energy_preds = np.concatenate(energy_preds)
    energy_trues = np.concatenate(energy_trues)

    metrics = {}

    energy_mse = np.mean((energy_preds - energy_trues) ** 2)
    energy_rmse = np.sqrt(energy_mse)
    energy_mae = np.mean(np.abs(energy_preds - energy_trues))

    metrics["energy_rmse"] = energy_rmse
    metrics["energy_mae"] = energy_mae
    metrics["energy_mse"] = energy_mse

    if len(force_preds) > 0:
        force_preds = np.concatenate(force_preds)
        force_trues = np.concatenate(force_trues)

        force_mse = np.mean((force_preds - force_trues) ** 2)
        force_rmse = np.sqrt(force_mse)
        force_mae = np.mean(np.abs(force_preds - force_trues))

        metrics["force_rmse"] = force_rmse
        metrics["force_mae"] = force_mae
        metrics["force_mse"] = force_mse

        force_errors = np.linalg.norm(force_preds - force_trues, axis=-1)
        metrics["force_magnitude_rmse"] = np.sqrt(np.mean(force_errors**2))
        metrics["force_magnitude_mae"] = np.mean(np.abs(force_errors))

    metrics["predictions"] = {
        "energy_pred": energy_preds,
        "energy_true": energy_trues,
    }

    if len(force_preds) > 0:
        metrics["predictions"]["force_pred"] = force_preds
        metrics["predictions"]["force_true"] = force_trues

    return metrics


def plot_energy_predictions(energy_pred, energy_true, output_path):
    """Plot predicted vs true energies.

    Args:
        energy_pred: Predicted energies
        energy_true: True energies
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(energy_true, energy_pred, alpha=0.6, s=20)

    min_val = min(energy_true.min(), energy_pred.min())
    max_val = max(energy_true.max(), energy_pred.max())

    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")

    rmse = np.sqrt(np.mean((energy_pred - energy_true) ** 2))
    mae = np.mean(np.abs(energy_pred - energy_true))

    ax.set_xlabel("True Energy", fontsize=12)
    ax.set_ylabel("Predicted Energy", fontsize=12)
    ax.set_title(f"Energy Predictions\nRMSE: {rmse:.6f}, MAE: {mae:.6f}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_force_predictions(force_pred, force_true, output_path):
    """Plot predicted vs true forces.

    Args:
        force_pred: Predicted forces [N_total, 3]
        force_true: True forces [N_total, 3]
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    components = ["Fx", "Fy", "Fz"]
    colors = ["r", "g", "b"]

    for i, (ax, comp, color) in enumerate(zip(axes, components, colors)):
        ax.scatter(force_true[:, i], force_pred[:, i], alpha=0.6, s=10, c=color)

        min_val = min(force_true[:, i].min(), force_pred[:, i].min())
        max_val = max(force_true[:, i].max(), force_pred[:, i].max())

        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

        rmse = np.sqrt(np.mean((force_pred[:, i] - force_true[:, i]) ** 2))
        mae = np.mean(np.abs(force_pred[:, i] - force_true[:, i]))

        ax.set_xlabel(f"True {comp}", fontsize=12)
        ax.set_ylabel(f"Predicted {comp}", fontsize=12)
        ax.set_title(f"{comp}\nRMSE: {rmse:.6f}, MAE: {mae:.6f}", fontsize=14)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_energy_error_histogram(energy_pred, energy_true, output_path):
    """Plot histogram of energy prediction errors.

    Args:
        energy_pred: Predicted energies
        energy_true: True energies
        output_path: Path to save plot
    """
    errors = energy_pred - energy_true
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)

    mean_error = np.mean(errors)
    std_error = np.std(errors)

    ax.axvline(mean_error, color="r", linestyle="--", linewidth=2, label=f"Mean: {mean_error:.6f}")
    ax.axvline(0, color="k", linestyle="-", linewidth=1, label="Zero")

    ax.set_xlabel("Prediction Error (Pred - True)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Energy Error Distribution\nStd: {std_error:.6f}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_force_error_histogram(force_pred, force_true, output_path):
    """Plot histogram of force prediction errors.

    Args:
        force_pred: Predicted forces [N_total, 3]
        force_true: True forces [N_total, 3]
        output_path: Path to save plot
    """
    errors = force_pred - force_true
    error_magnitudes = np.linalg.norm(errors, axis=-1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(error_magnitudes, bins=50, edgecolor="black", alpha=0.7)

    mean_error = np.mean(error_magnitudes)
    std_error = np.std(error_magnitudes)

    ax.axvline(mean_error, color="r", linestyle="--", linewidth=2, label=f"Mean: {mean_error:.6f}")

    ax.set_xlabel("Force Magnitude Error", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Force Magnitude Error Distribution\nStd: {std_error:.6f}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_metrics(metrics):
    """Print evaluation metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
    """
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(f"\nEnergy Metrics:")
    print(f"  RMSE: {metrics['energy_rmse']:.6f}")
    print(f"  MAE:  {metrics['energy_mae']:.6f}")
    print(f"  MSE:  {metrics['energy_mse']:.6f}")

    if "force_rmse" in metrics:
        print(f"\nForce Metrics (component-wise):")
        print(f"  RMSE: {metrics['force_rmse']:.6f}")
        print(f"  MAE:  {metrics['force_mae']:.6f}")
        print(f"  MSE:  {metrics['force_mse']:.6f}")

        print(f"\nForce Magnitude Metrics:")
        print(f"  RMSE: {metrics['force_magnitude_rmse']:.6f}")
        print(f"  MAE:  {metrics['force_magnitude_mae']:.6f}")

    print("=" * 60)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    dataset = XYZDataset(args.dataset)

    print(f"Dataset loaded: {len(dataset)} systems")

    # checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_kwargs = yaml.safe_load(f)

    if args.use_kan:
        model = MLPPotentialWithKANXYZ(
            n_atom_types=dataset.n_atom_types,
            hidden_dim=config_kwargs["hidden_dim"],
            n_edge_layers=config_kwargs["n_edge_layers"],
            n_node_layers=config_kwargs["n_node_layers"],
            atom_embedding_dim=config_kwargs["atom_embedding_dim"],
        )
    else:
        model = MLPPotentialXYZ(
            n_atom_types=dataset.n_atom_types,
            hidden_dim=config_kwargs["hidden_dim"],
            n_edge_layers=config_kwargs["n_edge_layers"],
            n_node_layers=config_kwargs["n_node_layers"],
            atom_embedding_dim=config_kwargs["atom_embedding_dim"],
        )

    # model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"Evaluating on {len(dataset)} systems")

    metrics = evaluate_model(
        model,
        dataloader,
        dataset,
        args.lambda_energy,
        args.lambda_force,
        device,
    )

    print_metrics(metrics)

    os.makedirs(args.output_dir, exist_ok=True)

    plot_energy_predictions(
        metrics["predictions"]["energy_pred"],
        metrics["predictions"]["energy_true"],
        os.path.join(args.output_dir, "energy_predictions.png"),
    )

    plot_energy_error_histogram(
        metrics["predictions"]["energy_pred"],
        metrics["predictions"]["energy_true"],
        os.path.join(args.output_dir, "energy_error_histogram.png"),
    )

    if "force_pred" in metrics["predictions"]:
        plot_force_predictions(
            metrics["predictions"]["force_pred"],
            metrics["predictions"]["force_true"],
            os.path.join(args.output_dir, "force_predictions.png"),
        )

        plot_force_error_histogram(
            metrics["predictions"]["force_pred"],
            metrics["predictions"]["force_true"],
            os.path.join(args.output_dir, "force_error_histogram.png"),
        )

    np.savez(
        os.path.join(args.output_dir, "metrics.npz"),
        **{k: v for k, v in metrics.items() if k != "predictions"},
    )

    np.savez(
        os.path.join(args.output_dir, "predictions.npz"),
        **metrics["predictions"],
    )

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
