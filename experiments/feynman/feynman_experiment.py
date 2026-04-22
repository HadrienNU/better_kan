"""
Main Feynman KAN experiment script.

This script trains KAN models on all Feynman analytical functions and evaluates
their performance across different KAN architectures and configurations.
"""

import os
import sys
import json
import time
import torch
import numpy as np
import argparse
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from better_kan import KAN, build_KAN
from better_kan.functions import Splines, ChebyshevPolynomial, RBFFunction
from better_kan.utils import train
from feynman_utils import load_feynman_function, create_feynman_dataset_from_id, evaluate_model, plot_function_comparison, plot_multi_model_comparison, save_results, create_experiment_summary


class FeynmanExperiment:
    """
    Main experiment class for training KANs on Feynman functions.
    """

    def __init__(self, results_dir: str = "results", train_samples: int = 1000, test_samples: int = 1000, training_steps: int = 100, noise_std: float = 0.0):
        """
        Initialize experiment configuration.

        Args:
            results_dir: Directory to save results
            train_samples: Number of training samples per function
            test_samples: Number of test samples per function
            training_steps: Number of training steps
            noise_std: Standard deviation of noise to add
        """
        # Auto-detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.results_dir = results_dir
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.training_steps = training_steps
        self.noise_std = noise_std

        # Create subdirectories
        os.makedirs(f"{results_dir}/losses", exist_ok=True)
        os.makedirs(f"{results_dir}/models", exist_ok=True)
        os.makedirs(f"{results_dir}/plots", exist_ok=True)
        os.makedirs(f"{results_dir}/logs", exist_ok=True)

        # Define model configurations to test
        self.model_configs = [
            {"name": "Splines_5", "layers_types": Splines, "grid_size": 5, "k": 3},
            {"name": "Splines_10", "layers_types": Splines, "grid_size": 10, "k": 3},
            {"name": "Splines_20", "layers_types": Splines, "grid_size": 20, "k": 3},
            {"name": "Chebyshev_3", "layers_types": ChebyshevPolynomial, "order": 3},
            {"name": "Chebyshev_5", "layers_types": ChebyshevPolynomial, "order": 5},
            {"name": "Chebyshev_7", "layers_types": ChebyshevPolynomial, "order": 7},
            {"name": "RBF_5", "layers_types": RBFFunction, "grid_size": 5},
            {"name": "RBF_10", "layers_types": RBFFunction, "grid_size": 10},
            {"name": "RBF_20", "layers_types": RBFFunction, "grid_size": 20},
        ]

    def create_model(self, config: Dict[str, Any], n_vars: int) -> KAN:
        """
        Create a KAN model based on configuration.

        Args:
            config: Model configuration dictionary
            n_vars: Number of input variables

        Returns:
            Configured KAN model
        """
        layers_types = config["layers_types"]

        # Build model based on configuration
        if "grid_size" in config:
            # For Splines and RBF
            if layers_types == RBFFunction:
                # RBF doesn't take 'k' parameter
                model = build_KAN(layers_types=layers_types, layers_hidden=[n_vars, 2 * n_vars + 1, 1], grid_size=config["grid_size"])
            else:
                # Splines take 'k' parameter
                model = build_KAN(layers_types=layers_types, layers_hidden=[n_vars, 2 * n_vars + 1, 1], grid_size=config["grid_size"], k=config.get("k", 3))
        elif "order" in config:

            model = build_KAN(layers_types=layers_types, layers_hidden=[n_vars, 2 * n_vars + 1, 1], poly_order=config["order"])
        else:
            # Default configuration
            model = build_KAN(layers_types=layers_types, layers_hidden=[n_vars, 2 * n_vars + 1, 1])

        # Move model to device
        model.to(self.device)
        return model

    def train_single_model(
        self, model: KAN, train_input: torch.Tensor, train_output: torch.Tensor, test_input: torch.Tensor, test_output: torch.Tensor, function_id: int, config_name: str
    ) -> Dict[str, Any]:
        """
        Train a single model and return results.

        Args:
            model: KAN model to train
            train_input: Training input data
            train_output: Training target data
            function_id: Function ID for logging
            config_name: Model configuration name

        Returns:
            Training results dictionary
        """
        # Train model
        start_time = time.time()

        try:
            # Create dataset dictionary for train function
            dataset = {"train_input": train_input, "train_label": train_output, "test_input": test_input, "test_label": test_output}

            results = train(model, dataset, opt="LBFGS", steps=self.training_steps, lamb=1e-5, update_grid=True, grid_update_freq=25)

            training_time = time.time() - start_time

            # Extract training losses and ensure they are regular Python floats
            raw_train_losses = results.get("train_loss", [])
            train_losses = []
            for loss in raw_train_losses:
                if hasattr(loss, "item"):
                    train_losses.append(float(loss.item()))
                elif isinstance(loss, (np.ndarray, np.generic)):
                    train_losses.append(float(loss))
                else:
                    train_losses.append(float(loss))

            return {"success": True, "train_losses": train_losses, "training_time": training_time, "final_model": model}

        except Exception as e:
            print(f"Training failed for function {function_id}, config {config_name}: {e}")
            return {"success": False, "error": str(e), "train_losses": [], "training_time": time.time() - start_time, "final_model": None}

    def evaluate_single_function(self, function_id: int) -> List[Dict[str, Any]]:
        """
        Train and evaluate all model configurations on a single function.

        Args:
            function_id: ID of the function to test

        Returns:
            List of results for all model configurations
        """
        print(f"\n=== Processing Function {function_id} ===")

        # Load function data
        try:
            function_data = load_feynman_function(function_id)
            print(f"Function: {function_data['name']}")
            print(f"Formula: {function_data['formula']}")
            print(f"Variables: {function_data['variables']}")
        except Exception as e:
            print(f"Failed to load function {function_id}: {e}")
            return []

        # Create dataset
        try:
            train_input, train_output, test_input, test_output = create_feynman_dataset_from_id(
                function_id, train_samples=self.train_samples, test_samples=self.test_samples, noise_std=self.noise_std, device=self.device
            )
        except Exception as e:
            print(f"Failed to create dataset for function {function_id}: {e}")
            return []

        n_vars = train_input.shape[1]
        results = []
        trained_models = []  # Collect trained models for multi-model plotting

        # Test each model configuration
        for config in self.model_configs:
            print(f"\n  Testing {config['name']}...")

            # Create model
            try:
                model = self.create_model(config, n_vars)
            except Exception as e:
                print(f"    Failed to create model: {e}")
                continue

            # Train model
            train_result = self.train_single_model(model, train_input, train_output, test_input, test_output, function_id, config["name"])

            if not train_result["success"]:
                print(f"    Training failed: {train_result.get('error', 'Unknown error')}")
                continue

            # Evaluate model
            try:
                metrics = evaluate_model(model, test_input, test_output, train_result["train_losses"])

                # Create result entry
                result = {
                    "function_id": function_id,
                    "function_name": function_data["name"],
                    "function_formula": function_data["formula"],
                    "model_type": config["name"],
                    "config": config,
                    "metrics": metrics,
                    "train_losses": train_result["train_losses"],
                    "training_time": train_result["training_time"],
                    "n_variables": n_vars,
                    "n_train_samples": self.train_samples,
                    "n_test_samples": self.test_samples,
                }

                results.append(result)

                # Store model for multi-model plotting
                trained_models.append({"model": model, "name": config["name"], "metrics": metrics})

                # Save model checkpoint
                model_path = f"{self.results_dir}/models/function_{function_id}_{config['name']}.pt"
                torch.save(model.state_dict(), model_path)

                # Save training losses
                if train_result["train_losses"]:
                    np.savetxt(f"{self.results_dir}/losses/function_{function_id}_{config['name']}.dat", train_result["train_losses"])

                print(f"    RMSE: {metrics['test_rmse']:.6f}, R²: {metrics['test_r2']:.6f}")

            except Exception as e:
                print(f"    Evaluation failed: {e}")
                print(traceback.format_exc())
                continue

        # Generate multi-model comparison plot
        if trained_models:
            self._create_multi_model_plot(trained_models, function_data, test_input, test_output, function_id)

        return results

    def _create_multi_model_plot(self, trained_models: List[Dict], function_data: Dict, test_input: torch.Tensor, test_output: torch.Tensor, function_id: int) -> None:
        """
        Create multi-model comparison plot for a single function.

        Args:
            trained_models: List of trained models with metadata
            function_data: Function metadata
            test_input: Test input data
            test_output: Test target values
            function_id: Function ID
        """
        models = [item["model"] for item in trained_models]
        model_names = [item["name"] for item in trained_models]

        plot_path = f"{self.results_dir}/plots/function_{function_id}_all_models.png"
        plot_multi_model_comparison(models, model_names, function_data, test_input, test_output, plot_path)

        print(f"  Multi-model comparison plot saved to {plot_path}")

    def run_full_experiment(self, start_function: int = 0, end_function: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run the complete experiment on all Feynman functions.

        Args:
            start_function: Starting function ID (inclusive)
            end_function: Ending function ID (exclusive), None for all functions

        Returns:
            List of all results
        """
        print("Starting Feynman KAN Experiment")
        print(f"Device: {self.device} (auto-detected)")
        print(f"Training samples: {self.train_samples}")
        print(f"Test samples: {self.test_samples}")
        print(f"Training steps: {self.training_steps}")
        print(f"Model configurations: {len(self.model_configs)}")

        # Determine function range - test with a reasonable number
        total_functions = 100  # Conservative estimate
        print(f"Note: Testing up to {total_functions} functions (will stop if function not found)")

        if end_function is None:
            end_function = total_functions

        end_function = min(end_function, total_functions)

        print(f"Testing functions {start_function} to {end_function-1} (total: {end_function-start_function})")

        all_results = []

        # Run experiment on each function
        for function_id in tqdm(range(start_function, end_function), desc="Processing functions"):
            results = self.evaluate_single_function(function_id)
            all_results.extend(results)

            # Save intermediate results
            if len(all_results) % 10 == 0:
                self.save_all_results(all_results, f"intermediate_{function_id}")

        # Create and save summary
        summary = create_experiment_summary(all_results)

        # Save final results
        self.save_all_results(all_results, "final")
        self.save_summary(summary)

        print(f"\n=== Experiment Complete ===")
        print(f"Total results: {len(all_results)}")
        print(f"Success rate: {summary.get('success_rate', {}).get('percentage', 0):.1f}%")
        print(f"Mean RMSE: {summary.get('performance_stats', {}).get('rmse_mean', 0):.6f}")
        print(f"Mean R²: {summary.get('performance_stats', {}).get('r2_mean', 0):.6f}")

        return all_results

    def save_all_results(self, results: List[Dict[str, Any]], suffix: str = "results") -> None:
        """
        Save all results to JSON file.

        Args:
            results: List of result dictionaries
            suffix: Suffix for the filename
        """
        results_path = f"{self.results_dir}/feynman_results_{suffix}.json"
        save_results(results, results_path)
        print(f"Saved results to {results_path}")

    def save_summary(self, summary: Dict[str, Any]) -> None:
        """
        Save experiment summary.

        Args:
            summary: Summary statistics dictionary
        """
        summary_path = f"{self.results_dir}/experiment_summary.json"
        save_results(summary, summary_path)
        print(f"Saved summary to {summary_path}")


def main():
    """Main function for running the experiment."""
    parser = argparse.ArgumentParser(description="Run Feynman KAN Experiment")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for training")
    parser.add_argument("--train_samples", type=int, default=1000, help="Number of training samples per function")
    parser.add_argument("--test_samples", type=int, default=1000, help="Number of test samples per function")
    parser.add_argument("--training_steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Standard deviation of noise to add")
    parser.add_argument("--start_function", type=int, default=0, help="Starting function ID")
    parser.add_argument("--end_function", type=int, default=None, help="Ending function ID (exclusive)")

    args = parser.parse_args()

    # Create experiment
    experiment = FeynmanExperiment(results_dir=args.results_dir, train_samples=args.train_samples, test_samples=args.test_samples, training_steps=args.training_steps, noise_std=args.noise_std)

    # Run experiment
    results = experiment.run_full_experiment(start_function=args.start_function, end_function=args.end_function)

    print(f"Experiment completed with {len(results)} total results")


if __name__ == "__main__":
    main()
