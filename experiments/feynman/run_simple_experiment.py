"""
Simple Feynman KAN experiment script.

Run a single KAN model on a single Feynman function.
"""

import os
import sys
import argparse
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from better_kan import KAN, build_KAN
from better_kan.functions import Splines, ChebyshevPolynomial, RBFFunction
from better_kan.utils import train
from feynman_utils import load_feynman_function, create_feynman_dataset_from_id, evaluate_model, plot_function_comparison, save_results


def run_simple_experiment(
    function_id: int,
    model_type: str = "Splines_10",
    train_samples: int = 1000,
    test_samples: int = 1000,
    training_steps: int = 100,
    noise_std: float = 0.0,
    results_dir: str = "results",
) -> dict:
    """
    Run a simple experiment: train one KAN model on one Feynman function.

    Args:
        function_id: ID of the Feynman function to fit
        model_type: Model configuration name (e.g., "Splines_10", "Chebyshev_5")
        train_samples: Number of training samples
        test_samples: Number of test samples
        training_steps: Number of training steps
        noise_std: Standard deviation of noise to add
        results_dir: Directory to save results

    Returns:
        Dictionary containing experiment results
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Function ID: {function_id}")
    print(f"Model type: {model_type}")
    print(f"Training samples: {train_samples}")
    print(f"Test samples: {test_samples}")
    print(f"Training steps: {training_steps}")

    function_data = load_feynman_function(function_id)
    print(f"\nFunction: {function_data['name']}")
    print(f"Formula: {function_data['formula']}")
    print(f"Variables: {function_data['variables']}")

    train_input, train_output, test_input, test_output = create_feynman_dataset_from_id(function_id, train_samples=train_samples, test_samples=test_samples, noise_std=noise_std, device=device)

    n_vars = train_input.shape[1]
    print(f"Number of variables: {n_vars}")

    dataset = {"train_input": train_input, "train_label": train_output, "test_input": test_input, "test_label": test_output}

    model_config = get_model_config(model_type)
    model = create_model(model_config, n_vars, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model_type}")
    print(f"Parameters: {n_params}")

    print("\nTraining...")
    results = train(model, dataset, opt="LBFGS", steps=training_steps, lamb=1e-5, update_grid=True, grid_update_freq=25)

    metrics = evaluate_model(model, test_input, test_output, results.get("train_loss", []))

    print(f"\nResults:")
    print(f"  Test RMSE: {metrics['test_rmse']:.6f}")
    print(f"  Test R²: {metrics['test_r2']:.6f}")
    print(f"  Test MAE: {metrics['test_mae']:.6f}")
    print(f"  Parameters: {metrics['n_parameters']}")

    result = {
        "function_id": function_id,
        "function_name": function_data["name"],
        "function_formula": function_data["formula"],
        "model_type": model_type,
        "config": model_config,
        "metrics": metrics,
        "train_losses": results.get("train_loss", []),
        "training_time": results.get("training_time", 0),
        "n_variables": n_vars,
        "n_train_samples": train_samples,
        "n_test_samples": test_samples,
    }

    results_path = f"{results_dir}/simple_experiment_results.json"
    save_results(result, results_path)
    print(f"\nResults saved to {results_path}")

    plot_path = f"{results_dir}/function_{function_id}_{model_type}.png"
    plot_function_comparison(model, function_data, test_input, test_output, plot_path)
    print(f"Plot saved to {plot_path}")

    return result


def get_model_config(model_type: str) -> dict:
    """Get model configuration by name."""
    configs = {
        "Splines_10": {"layers_types": Splines, "grid_size": 10, "k": 3, "name": "Splines_10"},
        "Splines_20": {"layers_types": Splines, "grid_size": 20, "k": 3, "name": "Splines_20"},
        "Splines_50": {"layers_types": Splines, "grid_size": 50, "k": 3, "name": "Splines_50"},
        "Chebyshev_3": {"layers_types": ChebyshevPolynomial, "order": 3, "name": "Chebyshev_3"},
        "Chebyshev_5": {"layers_types": ChebyshevPolynomial, "order": 5, "name": "Chebyshev_5"},
        "Chebyshev_7": {"layers_types": ChebyshevPolynomial, "order": 7, "name": "Chebyshev_7"},
        "RBF_10": {"layers_types": RBFFunction, "grid_size": 10, "name": "RBF_10"},
        "RBF_20": {"layers_types": RBFFunction, "grid_size": 20, "name": "RBF_20"},
        "RBF_50": {"layers_types": RBFFunction, "grid_size": 50, "name": "RBF_50"},
    }
    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(configs.keys())}")
    return configs[model_type]


def create_model(config: dict, n_vars: int, device: str) -> KAN:
    """Create a KAN model based on configuration."""
    layers_types = config["layers_types"]

    if "grid_size" in config:
        if layers_types == RBFFunction:
            model = build_KAN(layers_types=layers_types, layers_hidden=[n_vars, 2 * n_vars + 1, 1], grid_size=config["grid_size"])
        else:
            model = build_KAN(layers_types=layers_types, layers_hidden=[n_vars, 2 * n_vars + 1, 1], grid_size=config["grid_size"], k=config.get("k", 3))
    elif "order" in config:
        model = build_KAN(layers_types=layers_types, layers_hidden=[n_vars, 2 * n_vars + 1, 1], poly_order=config["order"])
    else:
        model = build_KAN(layers_types=layers_types, layers_hidden=[n_vars, 2 * n_vars + 1, 1])

    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Run Simple Feynman KAN Experiment")
    parser.add_argument("--function_id", type=int, default=1, help="Feynman function ID to fit")
    parser.add_argument("--model_type", type=str, default="Splines_10", help="Model type (e.g., Splines_10, Chebyshev_5)")
    parser.add_argument("--train_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--test_samples", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--training_steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Standard deviation of noise")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")

    args = parser.parse_args()

    run_simple_experiment(
        function_id=args.function_id,
        model_type=args.model_type,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        training_steps=args.training_steps,
        noise_std=args.noise_std,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
