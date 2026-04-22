"""
Analysis and visualization module for Feynman KAN experiment results.

This module provides functions to generate comprehensive plots, statistics,
and reports from the experiment results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any

from feynman_utils import load_results, create_experiment_summary


def set_plot_style():
    """Set consistent plot style for all visualizations."""
    plt.style.use("default")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.dpi"] = 150


def load_and_process_results(results_path: str) -> pd.DataFrame:
    """
    Load experiment results and convert to pandas DataFrame for analysis.

    Args:
        results_path: Path to the results JSON file

    Returns:
        DataFrame with flattened results
    """
    results = load_results(results_path)

    if not results:
        return pd.DataFrame()

    # Flatten nested dictionaries
    flattened_results = []
    for result in results:
        flat_result = {
            "function_id": result.get("function_id"),
            "function_name": result.get("function_name"),
            "function_formula": result.get("function_formula"),
            "model_type": result.get("model_type"),
            "n_variables": result.get("n_variables"),
            "n_train_samples": result.get("n_train_samples"),
            "n_test_samples": result.get("n_test_samples"),
            "training_time": result.get("training_time"),
        }

        # Add metrics
        metrics = result.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            flat_result[f"metric_{metric_name}"] = metric_value

        # Add configuration info
        config = result.get("config", {})
        for config_key, config_value in config.items():
            flat_result[f"config_{config_key}"] = config_value

        flattened_results.append(flat_result)

    return pd.DataFrame(flattened_results)


def plot_model_comparison(df: pd.DataFrame, save_path: str) -> None:
    """
    Create comparison plots for different model types.

    Args:
        df: DataFrame with experiment results
        save_path: Path to save the plot
    """
    set_plot_style()

    if df.empty or "model_type" not in df.columns:
        print("No valid data for model comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("KAN Model Comparison on Feynman Functions", fontsize=16)

    # RMSE comparison
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x="model_type", y="metric_test_rmse", ax=ax1)
    ax1.set_title("Test RMSE by Model Type")
    ax1.set_ylabel("RMSE")
    ax1.tick_params(axis="x", rotation=45)

    # R² comparison
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x="model_type", y="metric_test_r2", ax=ax2)
    ax2.set_title("Test R² by Model Type")
    ax2.set_ylabel("R²")
    ax2.tick_params(axis="x", rotation=45)

    # Training time comparison
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x="model_type", y="training_time", ax=ax3)
    ax3.set_title("Training Time by Model Type")
    ax3.set_ylabel("Training Time (s)")
    ax3.tick_params(axis="x", rotation=45)

    # Model complexity comparison
    ax4 = axes[1, 1]
    sns.boxplot(data=df, x="model_type", y="metric_n_parameters", ax=ax4)
    ax4.set_title("Model Complexity by Model Type")
    ax4.set_ylabel("Number of Parameters")
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_performance_distribution(df: pd.DataFrame, save_path: str) -> None:
    """
    Plot distribution of performance metrics.

    Args:
        df: DataFrame with experiment results
        save_path: Path to save the plot
    """
    set_plot_style()

    if df.empty:
        print("No valid data for performance distribution")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Performance Metrics Distribution", fontsize=16)

    # RMSE distribution
    ax1 = axes[0, 0]
    if "metric_test_rmse" in df.columns:
        ax1.hist(df["metric_test_rmse"], bins=30, alpha=0.7, edgecolor="black")
        ax1.set_title("RMSE Distribution")
        ax1.set_xlabel("RMSE")
        ax1.set_ylabel("Frequency")
        ax1.axvline(df["metric_test_rmse"].mean(), color="red", linestyle="--", label=f'Mean: {df["metric_test_rmse"].mean():.4f}')
        ax1.legend()

    # R² distribution
    ax2 = axes[0, 1]
    if "metric_test_r2" in df.columns:
        ax2.hist(df["metric_test_r2"], bins=30, alpha=0.7, edgecolor="black")
        ax2.set_title("R² Distribution")
        ax2.set_xlabel("R²")
        ax2.set_ylabel("Frequency")
        ax2.axvline(df["metric_test_r2"].mean(), color="red", linestyle="--", label=f'Mean: {df["metric_test_r2"].mean():.4f}')
        ax2.legend()

    # MAE distribution
    ax3 = axes[1, 0]
    if "metric_test_mae" in df.columns:
        ax3.hist(df["metric_test_mae"], bins=30, alpha=0.7, edgecolor="black")
        ax3.set_title("MAE Distribution")
        ax3.set_xlabel("MAE")
        ax3.set_ylabel("Frequency")
        ax3.axvline(df["metric_test_mae"].mean(), color="red", linestyle="--", label=f'Mean: {df["metric_test_mae"].mean():.4f}')
        ax3.legend()

    # Success rate (R² > 0.9)
    ax4 = axes[1, 1]
    if "metric_test_r2" in df.columns:
        success_rate = (df["metric_test_r2"] > 0.9).mean() * 100
        failure_rate = 100 - success_rate
        ax4.pie([success_rate, failure_rate], labels=["Success (R² > 0.9)", "Failure (R² ≤ 0.9)"], colors=["green", "red"], autopct="%1.1f%%")
        ax4.set_title("Overall Success Rate")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_function_difficulty_analysis(df: pd.DataFrame, save_path: str) -> None:
    """
    Analyze which functions are most difficult to approximate.

    Args:
        df: DataFrame with experiment results
        save_path: Path to save the plot
    """
    set_plot_style()

    if df.empty or "function_id" not in df.columns:
        print("No valid data for function difficulty analysis")
        return

    # Calculate average performance per function
    function_stats = df.groupby(["function_id", "function_name"]).agg({"metric_test_rmse": ["mean", "std"], "metric_test_r2": ["mean", "std"], "metric_test_mae": ["mean", "std"]}).reset_index()

    # Flatten column names
    function_stats.columns = ["function_id", "function_name"] + [f"{col[0]}_{col[1]}" for col in function_stats.columns[2:]]

    # Sort by RMSE (higher is more difficult)
    function_stats = function_stats.sort_values("metric_test_rmse_mean", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle("Function Difficulty Analysis", fontsize=16)

    # Top 10 most difficult functions
    ax1 = axes[0]
    top_difficult = function_stats.head(10)
    y_pos = range(len(top_difficult))

    bars = ax1.barh(y_pos, top_difficult["metric_test_rmse_mean"])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{name[:30]}..." if len(name) > 30 else name for name in top_difficult["function_name"]], fontsize=8)
    ax1.set_xlabel("Mean RMSE")
    ax1.set_title("Top 10 Most Difficult Functions")
    ax1.invert_yaxis()

    # Add error bars
    ax1.errorbar(top_difficult["metric_test_rmse_mean"], y_pos, xerr=top_difficult["metric_test_rmse_std"], fmt="none", color="black", capsize=3, alpha=0.5)

    # Bottom 10 easiest functions
    ax2 = axes[1]
    bottom_easy = function_stats.tail(10)
    y_pos = range(len(bottom_easy))

    bars = ax2.barh(y_pos, bottom_easy["metric_test_rmse_mean"])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{name[:30]}..." if len(name) > 30 else name for name in bottom_easy["function_name"]], fontsize=8)
    ax2.set_xlabel("Mean RMSE")
    ax2.set_title("Top 10 Easiest Functions")

    # Add error bars
    ax2.errorbar(bottom_easy["metric_test_rmse_mean"], y_pos, xerr=bottom_easy["metric_test_rmse_std"], fmt="none", color="black", capsize=3, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_configuration_analysis(df: pd.DataFrame, save_path: str) -> None:
    """
    Analyze performance by configuration parameters.

    Args:
        df: DataFrame with experiment results
        save_path: Path to save the plot
    """
    set_plot_style()

    if df.empty:
        print("No valid data for configuration analysis")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Configuration Parameter Analysis", fontsize=16)

    # Filter for models with grid_size
    grid_models = df[df["config_grid_size"].notna()]

    if not grid_models.empty:
        # Grid size vs performance
        ax1 = axes[0, 0]
        grid_performance = grid_models.groupby("config_grid_size")["metric_test_rmse"].mean()
        ax1.plot(grid_performance.index, grid_performance.values, marker="o")
        ax1.set_xlabel("Grid Size")
        ax1.set_ylabel("Mean RMSE")
        ax1.set_title("Grid Size vs Performance")
        ax1.grid(True, alpha=0.3)

    # Filter for Chebyshev models with order
    chebyshev_models = df[df["config_order"].notna()]

    if not chebyshev_models.empty:
        # Order vs performance
        ax2 = axes[0, 1]
        order_performance = chebyshev_models.groupby("config_order")["metric_test_rmse"].mean()
        ax2.plot(order_performance.index, order_performance.values, marker="o")
        ax2.set_xlabel("Polynomial Order")
        ax2.set_ylabel("Mean RMSE")
        ax2.set_title("Polynomial Order vs Performance")
        ax2.grid(True, alpha=0.3)

    # Number of variables vs performance
    ax3 = axes[1, 0]
    if "n_variables" in df.columns:
        var_performance = df.groupby("n_variables")["metric_test_rmse"].mean()
        ax3.bar(var_performance.index, var_performance.values)
        ax3.set_xlabel("Number of Variables")
        ax3.set_ylabel("Mean RMSE")
        ax3.set_title("Number of Variables vs Performance")
        ax3.grid(True, alpha=0.3)

    # Training time vs performance
    ax4 = axes[1, 1]
    if "training_time" in df.columns and "metric_test_rmse" in df.columns:
        ax4.scatter(df["training_time"], df["metric_test_rmse"], alpha=0.6)
        ax4.set_xlabel("Training Time (s)")
        ax4.set_ylabel("Test RMSE")
        ax4.set_title("Training Time vs Performance")

        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df["training_time"], df["metric_test_rmse"], 1)
            p = np.poly1d(z)
            ax4.plot(sorted(df["training_time"]), p(sorted(df["training_time"])), "r--", alpha=0.8, label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
            ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_comprehensive_report(results_path: str, output_dir: str) -> None:
    """
    Generate a comprehensive analysis report with plots and statistics.

    Args:
        results_path: Path to the experiment results
        output_dir: Directory to save the report
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load and process results
    print("Loading and processing results...")
    df = load_and_process_results(results_path)

    if df.empty:
        print("No valid results to analyze")
        return

    # Generate summary statistics
    summary = create_experiment_summary(load_results(results_path))

    # Save summary to JSON
    summary_path = os.path.join(output_dir, "comprehensive_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Generate plots
    print("Generating model comparison plots...")
    plot_model_comparison(df, os.path.join(output_dir, "model_comparison.png"))

    print("Generating performance distribution plots...")
    plot_performance_distribution(df, os.path.join(output_dir, "performance_distribution.png"))

    print("Generating function difficulty analysis...")
    plot_function_difficulty_analysis(df, os.path.join(output_dir, "function_difficulty.png"))

    print("Generating configuration analysis...")
    plot_configuration_analysis(df, os.path.join(output_dir, "configuration_analysis.png"))

    # Generate detailed statistics table
    print("Generating detailed statistics...")
    stats_table = generate_statistics_table(df, summary)

    stats_path = os.path.join(output_dir, "detailed_statistics.txt")
    with open(stats_path, "w") as f:
        f.write(stats_table)

    # Generate model rankings
    print("Generating model rankings...")
    rankings = generate_model_rankings(df)

    rankings_path = os.path.join(output_dir, "model_rankings.json")
    with open(rankings_path, "w") as f:
        json.dump(rankings, f, indent=2)

    print(f"Comprehensive report generated in {output_dir}")
    print(f"- Summary: {summary_path}")
    print(f"- Statistics: {stats_path}")
    print(f"- Rankings: {rankings_path}")
    print(f"- Plots: {output_dir}/*.png")


def generate_statistics_table(df: pd.DataFrame, summary: Dict[str, Any]) -> str:
    """
    Generate a formatted statistics table.

    Args:
        df: DataFrame with experiment results
        summary: Summary statistics dictionary

    Returns:
        Formatted string with statistics
    """
    stats = []
    stats.append("=== FEYNMAN KAN EXPERIMENT STATISTICS ===\n")

    # Overall statistics
    stats.append("OVERALL STATISTICS:")
    stats.append(f"Total Results: {len(df)}")
    stats.append(f"Success Rate (R² > 0.9): {summary.get('success_rate', {}).get('percentage', 0):.1f}%")

    if "performance_stats" in summary:
        ps = summary["performance_stats"]
        stats.append(f"Mean RMSE: {ps.get('rmse_mean', 0):.6f} ± {ps.get('rmse_std', 0):.6f}")
        stats.append(f"Median RMSE: {ps.get('rmse_median', 0):.6f}")
        stats.append(f"Mean R²: {ps.get('r2_mean', 0):.6f} ± {ps.get('r2_std', 0):.6f}")
        stats.append(f"Median R²: {ps.get('r2_median', 0):.6f}")
        stats.append(f"Mean MAE: {ps.get('mae_mean', 0):.6f} ± {ps.get('mae_std', 0):.6f}")

    stats.append("")

    # Model type statistics
    if "model_types" in summary:
        stats.append("MODEL TYPE STATISTICS:")
        for model_type, stats_dict in summary["model_types"].items():
            stats.append(f"\n{model_type}:")
            stats.append(f"  Count: {stats_dict.get('count', 0)}")
            stats.append(f"  Mean RMSE: {stats_dict.get('rmse_mean', 0):.6f}")
            stats.append(f"  Mean R²: {stats_dict.get('r2_mean', 0):.6f}")
            stats.append(f"  Mean MAE: {stats_dict.get('mae_mean', 0):.6f}")

    stats.append("")

    # Function difficulty
    if not df.empty and "function_id" in df.columns:
        stats.append("FUNCTION DIFFICULTY:")
        function_stats = df.groupby("function_name")["metric_test_rmse"].agg(["mean", "std", "count"])
        function_stats = function_stats.sort_values("mean", ascending=False)

        stats.append("\nTop 10 Most Difficult Functions:")
        for i, (func_name, stats_row) in enumerate(function_stats.head(10).iterrows()):
            stats.append(f"  {i+1}. {func_name[:50]}...: {stats_row['mean']:.6f} ± {stats_row['std']:.6f}")

        stats.append("\nTop 10 Easiest Functions:")
        for i, (func_name, stats_row) in enumerate(function_stats.tail(10).iterrows()):
            stats.append(f"  {i+1}. {func_name[:50]}...: {stats_row['mean']:.6f} ± {stats_row['std']:.6f}")

    return "\n".join(stats)


def generate_model_rankings(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate rankings of models by different criteria.

    Args:
        df: DataFrame with experiment results

    Returns:
        Dictionary with model rankings
    """
    rankings = {}

    if df.empty or "model_type" not in df.columns:
        return rankings

    # Rank by RMSE (lower is better)
    rmse_ranking = df.groupby("model_type")["metric_test_rmse"].mean().sort_values()
    rankings["best_rmse"] = rmse_ranking.index.tolist()

    # Rank by R² (higher is better)
    r2_ranking = df.groupby("model_type")["metric_test_r2"].mean().sort_values(ascending=False)
    rankings["best_r2"] = r2_ranking.index.tolist()

    # Rank by training time (lower is better)
    if "training_time" in df.columns:
        time_ranking = df.groupby("model_type")["training_time"].mean().sort_values()
        rankings["fastest_training"] = time_ranking.index.tolist()

    # Rank by success rate
    success_rates = df.groupby("model_type").apply(lambda x: (x["metric_test_r2"] > 0.9).mean())
    success_ranking = success_rates.sort_values(ascending=False)
    rankings["highest_success_rate"] = success_ranking.index.tolist()

    # Overall score (normalized combination)
    model_scores = {}
    for model_type in df["model_type"].unique():
        model_data = df[df["model_type"] == model_type]

        # Normalize metrics (0-1, higher is better)
        rmse_score = 1 - (model_data["metric_test_rmse"] - df["metric_test_rmse"].min()) / (df["metric_test_rmse"].max() - df["metric_test_rmse"].min())
        r2_score = (model_data["metric_test_r2"] - df["metric_test_r2"].min()) / (df["metric_test_r2"].max() - df["metric_test_r2"].min())

        if "training_time" in df.columns:
            time_score = 1 - (model_data["training_time"] - df["training_time"].min()) / (df["training_time"].max() - df["training_time"].min())
            overall_score = (rmse_score.mean() + r2_score.mean() + time_score.mean()) / 3
        else:
            overall_score = (rmse_score.mean() + r2_score.mean()) / 2

        model_scores[model_type] = overall_score

    overall_ranking = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    rankings["overall_best"] = [model for model, score in overall_ranking]

    return rankings
