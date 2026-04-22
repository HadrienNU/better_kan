"""
Visualization tools for ML Potential training and results.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(history_path: str, output_path: str):
    """Plot training and validation loss curves.

    Args:
        history_path: Path to training history npz file
        output_path: Path to save plot
    """
    data = np.load(history_path)

    train_losses = data["train_losses"]
    val_losses = data["val_losses"]
    train_energy = data["train_energy"]
    val_energy = data["val_energy"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train", linewidth=2)
    axes[0].plot(val_losses, label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Total Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_energy, label="Train", linewidth=2)
    axes[1].plot(val_energy, label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Energy Loss", fontsize=12)
    axes[1].set_title("Energy Loss", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_learning_rate_schedule(output_path: str):
    """Plot learning rate schedule (placeholder).

    Args:
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, "Learning rate schedule visualization", ha="center", va="center", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_history(history_path: str, output_path: str):
    """Plot detailed loss history.

    Args:
        history_path: Path to training history npz file
        output_path: Path to save plot
    """
    data = np.load(history_path)

    train_losses = data["train_losses"]
    val_losses = data["val_losses"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_losses, label="Train", linewidth=2, alpha=0.8)
    ax.plot(val_losses, label="Validation", linewidth=2, alpha=0.8)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training History", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_atom_energy_error(predictions_path: str, dataset_info_path: str, output_path: str):
    """Plot per-atom energy error.

    Args:
        predictions_path: Path to predictions npz file
        dataset_info_path: Path to dataset info with n_atoms
        output_path: Path to save plot
    """
    predictions = np.load(predictions_path)

    energy_pred = predictions["energy_pred"]
    energy_true = predictions["energy_true"]

    energy_errors = np.abs(energy_pred - energy_true)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(energy_errors, bins=50, edgecolor="black", alpha=0.7)

    mean_error = np.mean(energy_errors)
    std_error = np.std(energy_errors)

    ax.axvline(mean_error, color="r", linestyle="--", linewidth=2, label=f"Mean: {mean_error:.6f}")

    ax.set_xlabel("Absolute Energy Error", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Energy Error Distribution\nStd: {std_error:.6f}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_force_components_histogram(predictions_path: str, output_path: str):
    """Plot histogram of force components.

    Args:
        predictions_path: Path to predictions npz file
        output_path: Path to save plot
    """
    predictions = np.load(predictions_path)

    force_pred = predictions["force_pred"]
    force_true = predictions["force_true"]

    force_errors = force_pred - force_true

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    components = ["Fx", "Fy", "Fz"]

    for i, (ax, comp) in enumerate(zip(axes, components)):
        ax.hist(force_errors[:, i], bins=50, edgecolor="black", alpha=0.7)

        mean_error = np.mean(force_errors[:, i])
        std_error = np.std(force_errors[:, i])

        ax.axvline(mean_error, color="r", linestyle="--", linewidth=2, label=f"Mean: {mean_error:.6f}")

        ax.set_xlabel(f"Error ({comp})", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{comp} Error Distribution\nStd: {std_error:.6f}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_summary_dashboard(history_path: str, metrics_path: str, predictions_path: str, output_dir: str):
    """Create a summary dashboard with multiple plots.

    Args:
        history_path: Path to training history npz file
        metrics_path: Path to metrics npz file
        predictions_path: Path to predictions npz file
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(history_path)
    metrics = np.load(metrics_path)
    predictions = np.load(predictions_path)

    fig = plt.figure(figsize=(16, 12))

    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    train_losses = data["train_losses"]
    val_losses = data["val_losses"]
    ax1.plot(train_losses, label="Train", linewidth=2)
    ax1.plot(val_losses, label="Validation", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.set_title("Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    train_energy = data["train_energy"]
    val_energy = data["val_energy"]
    ax2.plot(train_energy, label="Train", linewidth=2)
    ax2.plot(val_energy, label="Validation", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Energy Loss")
    ax2.set_title("Energy Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    energy_pred = predictions["energy_pred"]
    energy_true = predictions["energy_true"]
    ax3.scatter(energy_true, energy_pred, alpha=0.6, s=20)
    min_val = min(energy_true.min(), energy_pred.min())
    max_val = max(energy_true.max(), energy_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax3.set_xlabel("True Energy")
    ax3.set_ylabel("Predicted Energy")
    ax3.set_title("Energy Predictions")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    energy_errors = energy_pred - energy_true
    ax4.hist(energy_errors, bins=50, edgecolor="black", alpha=0.7)
    ax4.axvline(0, color="r", linestyle="--", linewidth=2)
    ax4.set_xlabel("Prediction Error")
    ax4.set_ylabel("Count")
    ax4.set_title("Energy Error Distribution")
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 0])
    components = ["Fx", "Fy", "Fz"]
    for i, comp in enumerate(components):
        if f"force_pred" in predictions:
            force_pred = predictions["force_pred"]
            force_true = predictions["force_true"]
            force_errors = force_pred[:, i] - force_true[:, i]
            ax5.hist(force_errors, bins=50, alpha=0.6, label=comp)
    ax5.legend()
    ax5.set_xlabel("Error")
    ax5.set_ylabel("Count")
    ax5.set_title("Force Component Errors")
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    text = "Summary Metrics\n\n"
    text += f"Energy RMSE: {metrics['energy_rmse']:.6f}\n"
    text += f"Energy MAE:  {metrics['energy_mae']:.6f}\n"
    if "force_rmse" in metrics:
        text += f"Force RMSE:  {metrics['force_rmse']:.6f}\n"
        text += f"Force MAE:   {metrics['force_mae']:.6f}\n"

    ax6.text(0.1, 0.5, text, fontsize=14, va="center")

    plt.savefig(os.path.join(output_dir, "summary_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ML Potential results")

    parser.add_argument("--history_path", type=str, default=None, help="Path to training history npz file")
    parser.add_argument("--metrics_path", type=str, default=None, help="Path to metrics npz file")
    parser.add_argument("--predictions_path", type=str, default=None, help="Path to predictions npz file")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory")

    parser.add_argument("--plot_curves", action="store_true", help="Plot training curves")
    parser.add_argument("--plot_dashboard", action="store_true", help="Create summary dashboard")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.plot_curves and args.history_path:
        plot_training_curves(
            args.history_path,
            os.path.join(args.output_dir, "training_curves.png"),
        )

        plot_loss_history(
            args.history_path,
            os.path.join(args.output_dir, "loss_history.png"),
        )

        print(f"Training curves saved to {args.output_dir}")

    if args.plot_dashboard:
        if not all([args.history_path, args.metrics_path, args.predictions_path]):
            print("Error: history_path, metrics_path, and predictions_path are required for dashboard")
            return

        create_summary_dashboard(
            args.history_path,
            args.metrics_path,
            args.predictions_path,
            args.output_dir,
        )

        print(f"Summary dashboard saved to {args.output_dir}")


if __name__ == "__main__":
    main()
