#!/usr/bin/env python3
"""
Command-line interface for running Feynman KAN experiments.

This script provides a convenient command-line interface to run the complete
Feynman experiment with various configuration options.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feynman_experiment import FeynmanExperiment
from feynman_analysis import generate_comprehensive_report


def validate_arguments(args):
    """Validate command line arguments."""
    # Check if results directory exists or can be created
    results_path = Path(args.results_dir)
    try:
        results_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create results directory {args.results_dir}: {e}")
        return False

    # Validate function range
    if args.start_function < 0:
        print("Error: start_function must be >= 0")
        return False

    if args.end_function is not None and args.end_function <= args.start_function:
        print("Error: end_function must be greater than start_function")
        return False

    # Validate sample sizes
    if args.train_samples <= 0 or args.test_samples <= 0:
        print("Error: train_samples and test_samples must be positive")
        return False

    # Validate training steps
    if args.training_steps <= 0:
        print("Error: training_steps must be positive")
        return False

    # Validate noise
    if args.noise_std < 0:
        print("Error: noise_std must be non-negative")
        return False

    return True


def run_experiment(args):
    """Run the experiment with given arguments."""
    print("=" * 60)
    print("FEYNMAN KAN EXPERIMENT")
    print("=" * 60)

    # Display configuration
    print(f"Results directory: {args.results_dir}")
    print(f"Device: auto-detected (CUDA available: {torch.cuda.is_available()})")
    print(f"Function range: {args.start_function} to {args.end_function-1 if args.end_function else 'end'}")
    print(f"Training samples: {args.train_samples}")
    print(f"Test samples: {args.test_samples}")
    print(f"Training steps: {args.training_steps}")
    print(f"Noise level: {args.noise_std}")
    print(f"Analysis only: {args.analysis_only}")
    print(f"Quick test: {args.quick_test}")
    print("=" * 60)

    # Create experiment
    experiment = FeynmanExperiment(results_dir=args.results_dir, train_samples=args.train_samples, test_samples=args.test_samples, training_steps=args.training_steps, noise_std=args.noise_std)

    if args.analysis_only:
        # Only run analysis on existing results
        print("Running analysis only...")
        results_files = list(Path(args.results_dir).glob("feynman_results_*.json"))

        if not results_files:
            print("Error: No results files found for analysis")
            return False

        # Use the most recent results file
        latest_results = max(results_files, key=os.path.getctime)
        print(f"Using results file: {latest_results}")

        # Generate comprehensive report
        report_dir = os.path.join(args.results_dir, "comprehensive_report")
        generate_comprehensive_report(str(latest_results), report_dir)

        print("Analysis complete!")
        return True

    else:
        # Run the experiment
        if args.quick_test:
            # Quick test mode: run only first 5 functions with reduced steps
            print("Running quick test (first 5 functions, 20 steps)...")
            results = experiment.run_full_experiment(start_function=0, end_function=5)
        else:
            # Full experiment
            results = experiment.run_full_experiment(start_function=args.start_function, end_function=args.end_function)

        if not results:
            print("Error: No results generated")
            return False

        # Generate comprehensive report if requested
        if args.generate_report:
            print("Generating comprehensive report...")
            report_dir = os.path.join(args.results_dir, "comprehensive_report")

            # Use the final results file
            results_file = os.path.join(args.results_dir, "feynman_results_final.json")
            if os.path.exists(results_file):
                generate_comprehensive_report(results_file, report_dir)
                print("Report generated!")
            else:
                print("Warning: Final results file not found, skipping report generation")

        return True


def print_examples():
    """Print usage examples."""
    print("\nUSAGE EXAMPLES:")
    print("=" * 50)

    print("\n1. Quick test (5 functions, 20 steps):")
    print("   python run_experiment.py --quick-test")

    print("\n2. Run full experiment:")
    print("   python run_experiment.py")

    print("\n3. Run with custom parameters:")
    print("   python run_experiment.py --training-steps 200 --noise-std 0.01")

    print("\n4. Run specific function range:")
    print("   python run_experiment.py --start-function 10 --end-function 20")

    print("\n5. Generate report from existing results:")
    print("   python run_experiment.py --analysis-only")

    print("\n6. Custom results directory:")
    print("   python run_experiment.py --results-dir /path/to/results --generate-report")

    print("\n7. Large dataset test:")
    print("   python run_experiment.py --train-samples 5000 --test-samples 2000")


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []

    try:
        import torch

        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")

    try:
        import matplotlib

        print(f"✓ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        missing_deps.append("matplotlib")

    try:
        import seaborn

        print(f"✓ Seaborn: {seaborn.__version__}")
    except ImportError:
        missing_deps.append("seaborn")

    try:
        import pandas

        print(f"✓ Pandas: {pandas.__version__}")
    except ImportError:
        missing_deps.append("pandas")

    try:
        import numpy

        print(f"✓ NumPy: {numpy.__version__}")
    except ImportError:
        missing_deps.append("numpy")

    try:
        import tqdm

        print(f"✓ TQDM: {tqdm.__version__}")
    except ImportError:
        missing_deps.append("tqdm")

    # Check better-kan
    try:
        from better_kan import KAN

        print("✓ better-kan: Available")
    except ImportError:
        missing_deps.append("better-kan (local)")

    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install missing dependencies before running the experiment.")
        return False

    # Check CUDA availability if requested
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA: Available ({torch.cuda.device_count()} devices)")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA: Not available")
    except:
        pass

    print()
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run Feynman KAN Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick-test                    # Quick test with 5 functions
  %(prog)s --device cuda                   # Run on GPU
  %(prog)s --start-function 10 --end-function 20  # Specific range
  %(prog)s --analysis-only                 # Generate report from existing results
  %(prog)s --train-samples 5000            # Larger dataset
        """,
    )

    # Basic experiment parameters
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results (default: results)")
    parser.add_argument("--training-steps", type=int, default=100, help="Number of training steps (default: 100)")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Standard deviation of noise to add (default: 0.0)")

    # Data parameters
    parser.add_argument("--train-samples", type=int, default=1000, help="Number of training samples per function (default: 1000)")
    parser.add_argument("--test-samples", type=int, default=1000, help="Number of test samples per function (default: 1000)")

    # Function range
    parser.add_argument("--start-function", type=int, default=0, help="Starting function ID (default: 0)")
    parser.add_argument("--end-function", type=int, default=None, help="Ending function ID (exclusive, default: all)")

    # Execution modes
    parser.add_argument("--quick-test", action="store_true", help="Run quick test (5 functions, 20 steps)")
    parser.add_argument("--analysis-only", action="store_true", help="Only analyze existing results")
    parser.add_argument("--generate-report", action="store_true", default=True, help="Generate comprehensive report after experiment")

    # Utility options
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies and exit")
    parser.add_argument("--examples", action="store_true", help="Show usage examples and exit")

    args = parser.parse_args()

    # Handle utility options
    if args.check_deps:
        check_dependencies()
        return

    if args.examples:
        print_examples()
        return

    # Check dependencies before running
    if not check_dependencies():
        sys.exit(1)

    # Override settings for quick test mode
    if args.quick_test:
        args.start_function = 0
        args.end_function = 5
        args.training_steps = 20
        args.train_samples = 500
        args.test_samples = 500

    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)

    # Change to experiment directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Run experiment
    try:
        success = run_experiment(args)
        if success:
            print("\n🎉 Experiment completed successfully!")
            print(f"Results saved in: {args.results_dir}")
        else:
            print("\n❌ Experiment failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠ Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
