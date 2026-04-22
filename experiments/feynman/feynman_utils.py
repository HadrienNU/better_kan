"""
Utility functions for Feynman KAN experiments.

This module provides helper functions for loading Feynman analytical functions,
creating datasets, evaluating models, and saving results.
"""

import traceback
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

# Import better-kan utilities
from better_kan.utils import create_dataset, plot
from better_kan import KAN


def load_feynman_function(function_id: int) -> Dict[str, Any]:
    """
    Load a specific Feynman analytical function by ID.

    Args:
        function_id: Index of the function to load (1-based indexing based on feynman.py)

    Returns:
        Dictionary containing function metadata and callable
    """
    try:
        import feynman

        # Use get_feynman_dataset with 1-based indexing
        symbols, expression, function, ranges = feynman.get_feynman_dataset(function_id)

        # Convert symbols to variable names
        variables = [str(symbol) for symbol in symbols]

        # Find overall domain (use first variable range as approximation)
        if ranges and len(ranges) > 0:
            if isinstance(ranges[0], list) and len(ranges[0]) == 2:
                domain = ranges[0]  # Use first variable's range
            else:
                domain = [-5, 5]  # Default domain
        else:
            domain = [-5, 5]

        return {
            "id": function_id,
            "name": f"Feynman_Function_{function_id}",
            "formula": str(expression),
            "expression": expression,
            "variables": variables,
            "function": function,
            "domain": domain,
            "noise_level": 0.0,
        }
    except Exception as e:
        raise ImportError(f"Could not load function {function_id}: {e}")


def create_feynman_dataset_from_id(
    function_id: int, train_samples: int = 1000, test_samples: int = 1000, noise_std: float = 0.0, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create training and test datasets for a Feynman function by ID.

    Args:
        function_id: Function ID (1-based)
        train_samples: Number of training samples
        test_samples: Number of test samples
        noise_std: Standard deviation of Gaussian noise to add
        device: Device to create tensors on

    Returns:
        Tuple of (train_input, train_output, test_input, test_output)
    """
    try:
        import feynman

        symbols, expression, func, ranges = feynman.get_feynman_dataset(function_id)
        n_vars = len(symbols)

        # Generate random samples based on variable ranges
        train_input = torch.zeros(train_samples, n_vars, device=device)
        test_input = torch.zeros(test_samples, n_vars, device=device)

        for var_idx, var_range in enumerate(ranges):
            if isinstance(var_range, list) and len(var_range) == 2:
                low, high = var_range
            else:
                low, high = -5, 5  # Default range

            train_input[:, var_idx] = torch.rand(train_samples, device=device) * (high - low) + low
            test_input[:, var_idx] = torch.rand(test_samples, device=device) * (high - low) + low

        # Compute function values
        train_output = torch.zeros(train_samples, 1, device=device)
        test_output = torch.zeros(test_samples, 1, device=device)

        # Try to evaluate function in batch first
        try:
            train_output = func(train_input)
            test_output = func(test_input)

            # Ensure proper shape
            if train_output.dim() == 1:
                train_output = train_output.unsqueeze(1)
            if test_output.dim() == 1:
                test_output = test_output.unsqueeze(1)

        except Exception as e:
            print(f"Batch evaluation failed: {e}, falling back to individual evaluation")
            # Fall back to individual evaluation
            train_output = torch.zeros(train_samples, 1, device=device)
            test_output = torch.zeros(test_samples, 1, device=device)

            for i in range(train_samples):
                try:
                    val = func(train_input[i])
                    train_output[i] = float(val)
                except Exception as e:
                    print(f"Warning: Function evaluation failed for training sample {i}: {e}")
                    train_output[i] = 0.0

            for i in range(test_samples):
                try:
                    val = func(test_input[i])
                    test_output[i] = float(val)
                except Exception as e:
                    print(f"Warning: Function evaluation failed for test sample {i}: {e}")
                    test_output[i] = 0.0

        # Add noise if specified
        if noise_std > 0:
            train_output += torch.randn_like(train_output) * noise_std
            test_output += torch.randn_like(test_output) * noise_std

        return train_input, train_output, test_input, test_output

    except Exception as e:
        raise RuntimeError(f"Failed to create dataset for function {function_id}: {e}")


def create_feynman_dataset(
    function_data: Dict[str, Any], train_samples: int = 1000, test_samples: int = 1000, noise_std: float = 0.0, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create training and test datasets for a Feynman function.
    This is a wrapper around create_feynman_dataset_from_id.

    Args:
        function_data: Function metadata from load_feynman_function
        train_samples: Number of training samples
        test_samples: Number of test samples
        noise_std: Standard deviation of Gaussian noise to add
        device: Device to create tensors on

    Returns:
        Tuple of (train_input, train_output, test_input, test_output)
    """
    return create_feynman_dataset_from_id(function_data["id"], train_samples, test_samples, noise_std, device)


def evaluate_model(model: KAN, test_input: torch.Tensor, test_output: torch.Tensor, train_losses: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Evaluate a trained KAN model on test data.

    Args:
        model: Trained KAN model
        test_input: Test input features
        test_output: Test target values
        train_losses: List of training losses (optional)

    Returns:
        Dictionary of evaluation metrics
    """
    try:
        model.eval()
        with torch.no_grad():
            predictions = model(test_input)

            # Compute RMSE
            mse = torch.mean((predictions - test_output) ** 2)
            rmse = float(torch.sqrt(mse.cpu()).item())

            # Compute R² score
            ss_res = torch.sum((test_output - predictions) ** 2)
            ss_tot = torch.sum((test_output - torch.mean(test_output)) ** 2)
            r2 = float((1 - ss_res.cpu() / ss_tot.cpu()).item())

            # Compute MAE
            mae = float(torch.mean(torch.abs(predictions - test_output)).cpu().item())

            # Model complexity (number of parameters)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            metrics = {
                "test_rmse": rmse,
                "test_r2": r2,
                "test_mae": mae,
                "n_parameters": n_params,
            }

            # Add training metrics if available
            if train_losses:
                # Convert any numpy arrays in train_losses to regular lists
                clean_train_losses = []
                for loss in train_losses:
                    if hasattr(loss, "item") and callable(getattr(loss, "item", None)):
                        clean_train_losses.append(float(loss.item()))
                    elif isinstance(loss, np.ndarray):
                        clean_train_losses.append(float(loss.flatten()[0]))
                    else:
                        clean_train_losses.append(float(loss))

                metrics["final_train_loss"] = clean_train_losses[-1] if clean_train_losses else float("inf")
                metrics["convergence_step"] = len(clean_train_losses)

                # Compute improvement
                if len(clean_train_losses) > 1:
                    initial_loss = clean_train_losses[0]
                    final_loss = clean_train_losses[-1]
                    metrics["loss_improvement"] = (initial_loss - final_loss) / initial_loss
                else:
                    metrics["loss_improvement"] = 0.0

            return metrics

    except Exception as e:
        print(f"Evaluation failed: {e}")
        print(traceback.format_exc())
        # Return empty metrics on failure
        return {"test_rmse": float("inf"), "test_r2": float("-inf"), "test_mae": float("inf"), "n_parameters": 0, "error": str(e)}


def plot_function_comparison(model: KAN, function_data: Dict[str, Any], test_input: torch.Tensor, test_output: torch.Tensor, save_path: str, n_plot_points: int = 200) -> None:
    """
    Generate comparison plot between true function and KAN approximation.

    Args:
        model: Trained KAN model
        function_data: Function metadata
        test_input: Test input data
        test_output: Test target values
        save_path: Path to save the plot
        n_plot_points: Number of points for plotting curve
    """
    model.eval()

    # Sort data for plotting
    if test_input.shape[1] == 1:  # 1D function
        sorted_indices = torch.argsort(test_input[:, 0])
        x_sorted = test_input[sorted_indices, 0].cpu().numpy()
        y_true = test_output[sorted_indices, 0].cpu().numpy()

        with torch.no_grad():
            y_pred = model(test_input[sorted_indices]).cpu().numpy().flatten()
        plt.figure(figsize=(10, 6))
        plt.plot(x_sorted, y_true, "b-", label="True Function", alpha=0.7, linewidth=2)
        plt.plot(x_sorted, y_pred, "r--", label="KAN Approximation", alpha=0.7, linewidth=2)
        plt.scatter(x_sorted[::10], y_true[::10], c="blue", s=10, alpha=0.3)

        plt.xlabel(function_data["variables"][0])
        plt.ylabel("Output")
        plt.title(f"Function {function_data['id']}: {function_data['name']}")
        plt.legend()
        plt.grid(True, alpha=0.3)

    else:  # Multi-dimensional function - plot predictions vs true
        with torch.no_grad():
            predictions = model(test_input).cpu().numpy().flatten()

        plt.figure(figsize=(10, 6))
        plt.scatter(test_output.cpu().numpy().flatten(), predictions, alpha=0.6)
        plt.plot([test_output.cpu().min(), test_output.cpu().max()], [test_output.cpu().min(), test_output.cpu().max()], "r--", lw=2)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Function {function_data['id']}: {function_data['name']}")
        plt.grid(True, alpha=0.3)

        # Add R² annotation
        ss_res = np.sum((test_output.cpu().numpy().flatten() - predictions) ** 2)
        ss_tot = np.sum((test_output.cpu().numpy().flatten() - np.mean(test_output.cpu().numpy().flatten())) ** 2)
        r2 = 1 - ss_res / ss_tot
        plt.annotate(f"R² = {r2:.4f}", xy=(0.05, 0.95), xycoords="axes fraction", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_multi_model_comparison(
    models: List[KAN], 
    model_names: List[str], 
    function_data: Dict[str, Any], 
    test_input: torch.Tensor, 
    test_output: torch.Tensor, 
    save_path: str, 
    n_plot_points: int = 200
) -> None:
    """
    Generate comparison plot between true function and multiple KAN approximations.
    
    Args:
        models: List of trained KAN models
        model_names: List of model configuration names
        function_data: Function metadata
        test_input: Test input data
        test_output: Test target values
        save_path: Path to save the plot
        n_plot_points: Number of points for plotting curve
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Define color scheme and line styles
    color_scheme = {
        'Splines': ['#1f77b4', '#ff7f0e', '#2ca02c'],  # Blues, oranges, greens
        'Chebyshev': ['#d62728', '#9467bd', '#8c564b'],  # Reds, purples, browns
        'RBF': ['#e377c2', '#7f7f7f', '#bcbd22']        # Pinks, grays, olives
    }
    
    line_styles = ['-', '--', '-.']
    
    plt.figure(figsize=(12, 8))
    
    # Calculate R² for each model for legend
    model_r2_scores = []
    for model in models:
        model.eval()
        with torch.no_grad():
            predictions = model(test_input).cpu().numpy().flatten()
        ss_res = np.sum((test_output.cpu().numpy().flatten() - predictions) ** 2)
        ss_tot = np.sum((test_output.cpu().numpy().flatten() - np.mean(test_output.cpu().numpy().flatten())) ** 2)
        r2 = 1 - ss_res / ss_tot
        model_r2_scores.append(r2)
    
    # Sort data for plotting (1D functions)
    if test_input.shape[1] == 1:  # 1D function
        sorted_indices = torch.argsort(test_input[:, 0])
        x_sorted = test_input[sorted_indices, 0].cpu().numpy()
        y_true = test_output[sorted_indices, 0].cpu().numpy()
        
        # Plot true function
        plt.plot(x_sorted, y_true, "b-", label="True Function", alpha=0.8, linewidth=3, zorder=10)
        plt.scatter(x_sorted[::20], y_true[::20], c="blue", s=15, alpha=0.4, zorder=9)
        
        # Plot each model's prediction
        for i, (model, name, r2) in enumerate(zip(models, model_names, model_r2_scores)):
            model.eval()
            with torch.no_grad():
                y_pred = model(test_input[sorted_indices]).cpu().numpy().flatten()
            
            # Determine color and style based on model type
            if 'Splines' in name:
                base_color = color_scheme['Splines'][int(name.split('_')[-1]) % 3]
            elif 'Chebyshev' in name:
                base_color = color_scheme['Chebyshev'][int(name.split('_')[-1]) % 3]
            elif 'RBF' in name:
                base_color = color_scheme['RBF'][int(name.split('_')[-1]) % 3]
            else:
                base_color = f'C{i}'  # Default color
            
            line_style = line_styles[i % len(line_styles)]
            
            plt.plot(x_sorted, y_pred, color=base_color, linestyle=line_style, 
                    label=f"{name} (R²={r2:.4f})", alpha=0.8, linewidth=2)
        
        plt.xlabel(function_data["variables"][0], fontsize=12)
        plt.ylabel("Output", fontsize=12)
        
    else:  # Multi-dimensional function - plot predictions vs true
        y_true_flat = test_output.cpu().numpy().flatten()
        
        # Plot reference diagonal line
        y_min, y_max = y_true_flat.min(), y_true_flat.max()
        plt.plot([y_min, y_max], [y_min, y_max], "k--", lw=2, alpha=0.7, label="Perfect Prediction")
        
        # Plot each model's predictions
        for i, (model, name, r2) in enumerate(zip(models, model_names, model_r2_scores)):
            model.eval()
            with torch.no_grad():
                predictions = model(test_input).cpu().numpy().flatten()
            
            # Determine color based on model type
            if 'Splines' in name:
                base_color = color_scheme['Splines'][int(name.split('_')[-1]) % 3]
            elif 'Chebyshev' in name:
                base_color = color_scheme['Chebyshev'][int(name.split('_')[-1]) % 3]
            elif 'RBF' in name:
                base_color = color_scheme['RBF'][int(name.split('_')[-1]) % 3]
            else:
                base_color = f'C{i}'  # Default color
            
            plt.scatter(y_true_flat, predictions, alpha=0.6, s=20, color=base_color, 
                       label=f"{name} (R²={r2:.4f})")
        
        plt.xlabel("True Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
    
    # Title and formatting
    title = f"Function {function_data['id']}: {function_data['name']}\n"
    if len(function_data.get('formula', '')) > 50:
        title += f"{function_data.get('formula', '')[:47]}..."
    else:
        title += function_data.get('formula', '')
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Enhanced legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for legend
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_results(results, save_path: str) -> None:
    """
    Save experiment results to JSON file.

    Args:
        results: Results dictionary or list to save
        save_path: Path to save the results
    """

    # Convert tensors to lists for JSON serialization
    def convert_tensors(obj):
        # Handle None
        if obj is None:
            return None
        # Handle basic types
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        # Handle tensors and arrays
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle numpy scalars
        elif hasattr(obj, 'dtype') and obj.dtype in [np.int32, np.int64, np.float32, np.float64]:
            return float(obj)
        # Handle collections
        elif isinstance(obj, dict):
            return {str(k): convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        elif isinstance(obj, tuple):
            return [convert_tensors(item) for item in obj]
        # Handle types and functions
        elif isinstance(obj, type):
            return f"<class '{obj.__name__}'>"
        elif callable(obj):
            return f"<function '{getattr(obj, '__name__', 'unknown')}'>"
        # Handle other objects by converting to string
        else:
            try:
                return str(obj)
            except:
                return f"<object of type '{type(obj).__name__}'>"

    try:
        serializable_results = convert_tensors(results)

        with open(save_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

    except Exception as e:
        print(f"Error in JSON serialization: {e}")

        # Debug: find problematic objects
        def find_problematic(obj, path="root"):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    try:
                        convert_tensors(v)
                    except Exception as e:
                        print(f"Problematic at {path}.{k}: {type(v)} - {e}")
                        find_problematic(v, f"{path}.{k}")
                    else:
                        find_problematic(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    try:
                        convert_tensors(item)
                    except Exception as e:
                        print(f"Problematic at {path}[{i}]: {type(item)} - {e}")
                        find_problematic(item, f"{path}[{i}]")

        print("Searching for problematic objects...")
        find_problematic(results)
        raise


def load_results(load_path: str) -> Dict[str, Any]:
    """
    Load experiment results from JSON file.

    Args:
        load_path: Path to load results from

    Returns:
        Results dictionary
    """
    with open(load_path, "r") as f:
        results = json.load(f)
    return results


def create_experiment_summary(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create summary statistics from all experiment results.

    Args:
        all_results: List of result dictionaries for all functions

    Returns:
        Summary statistics dictionary
    """
    summary = {"total_functions": len(all_results), "model_types": {}, "performance_stats": {}, "success_rate": {}}

    # Collect all metrics
    all_rmse = []
    all_r2 = []
    all_mae = []
    model_performance = {}

    for result in all_results:
        if "metrics" in result:
            metrics = result["metrics"]
            model_type = result.get("model_type", "unknown")

            if model_type not in model_performance:
                model_performance[model_type] = {"rmse": [], "r2": [], "mae": []}

            model_performance[model_type]["rmse"].append(metrics["test_rmse"])
            model_performance[model_type]["r2"].append(metrics["test_r2"])
            model_performance[model_type]["mae"].append(metrics["test_mae"])

            all_rmse.append(metrics["test_rmse"])
            all_r2.append(metrics["test_r2"])
            all_mae.append(metrics["test_mae"])

    # Compute statistics
    if all_rmse:
        summary["performance_stats"] = {
            "rmse_mean": np.mean(all_rmse),
            "rmse_std": np.std(all_rmse),
            "rmse_median": np.median(all_rmse),
            "r2_mean": np.mean(all_r2),
            "r2_std": np.std(all_r2),
            "r2_median": np.median(all_r2),
            "mae_mean": np.mean(all_mae),
            "mae_std": np.std(all_mae),
            "mae_median": np.median(all_mae),
        }

    # Model-specific statistics
    for model_type, metrics in model_performance.items():
        summary["model_types"][model_type] = {"count": len(metrics["rmse"]), "rmse_mean": np.mean(metrics["rmse"]), "r2_mean": np.mean(metrics["r2"]), "mae_mean": np.mean(metrics["mae"])}

    # Success rate (R² > 0.9)
    if all_r2:
        success_threshold = 0.9
        successful_fits = sum(1 for r2 in all_r2 if r2 > success_threshold)
        summary["success_rate"] = {"threshold": success_threshold, "count": successful_fits, "percentage": (successful_fits / len(all_r2)) * 100}

    return summary
