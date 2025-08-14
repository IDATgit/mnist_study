import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


def load_rsvd_data(model_name, data_type):
    """
    Load RSVD eigenvalue data for a specific model and data type.
    
    Args:
        model_name: Name of the model
        data_type: 'train' or 'test'
        
    Returns:
        eigenvalues: Array of eigenvalues (singular values from RSVD)
        stats: Dictionary of statistics
    """
    # Base directory for Fisher analysis outputs
    base_dir = Path(f'model_interpretation/outputs/fisher_analysis/{model_name}')
    
    if not base_dir.exists():
        raise FileNotFoundError(f"No Fisher analysis found for model: {model_name}")
    
    # Load RSVD singular values (eigenvalues)
    s_path = base_dir / f"{data_type}_{model_name}_S_rsvd.npy"
    
    if not s_path.exists():
        raise FileNotFoundError(f"No {data_type} RSVD eigenvalues found for model: {model_name}")
    
    print(f"Loading {data_type} RSVD eigenvalues for {model_name}")
    eigenvalues = np.load(s_path)
    
    # Load statistics
    stats_file = base_dir / f"{data_type}_{model_name}_fisher_stats_rsvd.txt"
    stats = {}
    
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    try:
                        stats[key.strip()] = float(value.strip())
                    except ValueError:
                        stats[key.strip()] = value.strip()
    
    return eigenvalues, stats


def compare_eigenvalue_spectra(train_eigenvals, test_eigenvals, model_name, output_dir):
    """
    Create comparison plots of eigenvalue spectra between train and test data.
    
    Args:
        train_eigenvals: Training eigenvalues
        test_eigenvals: Test eigenvalues
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overlaid eigenvalue spectra
    plt.figure(figsize=(12, 8))
    
    # Ensure both arrays have the same length for comparison
    min_len = min(len(train_eigenvals), len(test_eigenvals))
    train_subset = train_eigenvals[:min_len]
    test_subset = test_eigenvals[:min_len]
    
    indices = np.arange(1, min_len + 1)
    
    plt.plot(indices, train_subset, 'b-', linewidth=2, label='Train Data', alpha=0.8)
    plt.plot(indices, test_subset, 'r-', linewidth=2, label='Test Data', alpha=0.8)
    
    plt.title(f'Eigenvalue Spectrum Comparison (RSVD)\n{model_name}')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_train_vs_test_spectrum.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Eigenvalue ratio plot
    plt.figure(figsize=(12, 8))
    
    # Calculate ratio (train/test), avoiding division by zero
    ratio = np.divide(train_subset, test_subset, out=np.ones_like(train_subset), where=test_subset!=0)
    
    plt.plot(indices, ratio, 'g-', linewidth=2, alpha=0.8)
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Ratio = 1')
    
    plt.title(f'Eigenvalue Ratio (Train/Test) vs Index\n{model_name}')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue Ratio (Train/Test)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_train_test_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Cumulative eigenvalue comparison
    plt.figure(figsize=(12, 8))
    
    train_cumsum = np.cumsum(train_subset) / np.sum(train_subset) * 100
    test_cumsum = np.cumsum(test_subset) / np.sum(test_subset) * 100
    
    plt.plot(indices, train_cumsum, 'b-', linewidth=2, label='Train Data', alpha=0.8)
    plt.plot(indices, test_cumsum, 'r-', linewidth=2, label='Test Data', alpha=0.8)
    
    plt.title(f'Cumulative Eigenvalue Percentage\n{model_name}')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Cumulative Percentage (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_train_test_cumulative.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Difference plot
    plt.figure(figsize=(12, 8))
    
    diff = train_subset - test_subset
    
    plt.plot(indices, diff, 'purple', linewidth=2, alpha=0.8)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Difference = 0')
    
    plt.title(f'Eigenvalue Difference (Train - Test)\n{model_name}')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_train_test_difference.png', dpi=300, bbox_inches='tight')
    plt.close()


def compare_statistics(train_stats, test_stats, model_name, output_dir):
    """
    Compare and save statistical measures between train and test data.
    
    Args:
        train_stats: Training statistics dictionary
        test_stats: Test statistics dictionary
        model_name: Name of the model
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compare key statistics
    comparison_stats = {}
    
    for key in train_stats:
        if key in test_stats and isinstance(train_stats[key], (int, float)):
            train_val = train_stats[key]
            test_val = test_stats[key]
            
            comparison_stats[f'{key}_train'] = train_val
            comparison_stats[f'{key}_test'] = test_val
            comparison_stats[f'{key}_ratio_train_test'] = train_val / test_val if test_val != 0 else float('inf')
            comparison_stats[f'{key}_difference'] = train_val - test_val
    
    # Save comparison statistics
    with open(output_dir / f'{model_name}_train_test_comparison_stats.txt', 'w') as f:
        f.write(f"Statistical Comparison for {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in comparison_stats.items():
            if isinstance(value, float):
                f.write(f'{key}: {value:.6f}\n')
            else:
                f.write(f'{key}: {value}\n')
    
    return comparison_stats


def analyze_eigenvalue_correlation(train_eigenvals, test_eigenvals, model_name, output_dir):
    """
    Analyze correlation between train and test eigenvalues.
    
    Args:
        train_eigenvals: Training eigenvalues
        test_eigenvals: Test eigenvalues
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure same length
    min_len = min(len(train_eigenvals), len(test_eigenvals))
    train_subset = train_eigenvals[:min_len]
    test_subset = test_eigenvals[:min_len]
    
    # Calculate correlation
    correlation = np.corrcoef(train_subset, test_subset)[0, 1]
    
    # Scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(train_subset, test_subset, alpha=0.6, s=20)
    
    # Add diagonal line (perfect correlation)
    min_val = min(np.min(train_subset), np.min(test_subset))
    max_val = max(np.max(train_subset), np.max(test_subset))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Correlation')
    
    plt.xlabel('Train Eigenvalues')
    plt.ylabel('Test Eigenvalues')
    plt.title(f'Train vs Test Eigenvalue Correlation\n{model_name} (Correlation: {correlation:.4f})')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_train_test_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlation


def get_available_fisher_models():
    """
    Get list of models that have Fisher analysis data.
    
    Returns:
        List of model names that have Fisher analysis directories
    """
    fisher_base_dir = Path('model_interpretation/outputs/fisher_analysis')
    
    if not fisher_base_dir.exists():
        return []
    
    available_models = []
    for model_dir in fisher_base_dir.iterdir():
        if model_dir.is_dir():
            # Check if the directory has at least some RSVD files
            rsvd_files = list(model_dir.glob('*_S_rsvd.npy'))
            if rsvd_files:
                available_models.append(model_dir.name)
    
    return sorted(available_models)


def main():
    """
    Main function to compare train and test RSVD results for a selected model.
    """
    # Get available models that have Fisher analysis data
    available_models = get_available_fisher_models()
    
    if not available_models:
        print("No models with Fisher analysis data found!")
        print("Please run Fisher Information analysis first using fisher_information_rsvd.py")
        return
    
    # Display available models
    print("Available models for Fisher Information comparison:")
    for i, model_name in enumerate(available_models, 1):
        print(f"{i}. {model_name}")
    
    # Get user choice
    while True:
        try:
            choice = int(input(f"\nSelect model (1-{len(available_models)}): ")) - 1
            if 0 <= choice < len(available_models):
                selected_model = available_models[choice]
                break
            else:
                print(f"Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\nAnalyzing Fisher Information comparison for: {selected_model}")
    
    try:
        # Load train and test data
        train_eigenvals, train_stats = load_rsvd_data(selected_model, 'train')
        test_eigenvals, test_stats = load_rsvd_data(selected_model, 'test')
        
        print(f"Loaded train eigenvalues: {len(train_eigenvals)} components")
        print(f"Loaded test eigenvalues: {len(test_eigenvals)} components")
        
        # Create output directory
        output_dir = Path(f'model_interpretation/outputs/fisher_analysis/{selected_model}/train_test_comparison')
        
        # Generate comparison plots
        print("Generating comparison plots...")
        compare_eigenvalue_spectra(train_eigenvals, test_eigenvals, selected_model, output_dir)
        
        # Compare statistics
        print("Comparing statistics...")
        comparison_stats = compare_statistics(train_stats, test_stats, selected_model, output_dir)
        
        # Analyze correlation
        print("Analyzing eigenvalue correlation...")
        correlation = analyze_eigenvalue_correlation(train_eigenvals, test_eigenvals, selected_model, output_dir)
        
        # Print summary
        print(f"\nComparison Summary for {selected_model}:")
        print(f"Train eigenvalue correlation: {correlation:.4f}")
        print(f"Max train eigenvalue: {np.max(train_eigenvals):.6f}")
        print(f"Max test eigenvalue: {np.max(test_eigenvals):.6f}")
        print(f"Min train eigenvalue: {np.min(train_eigenvals):.6f}")
        print(f"Min test eigenvalue: {np.min(test_eigenvals):.6f}")
        
        if 'condition_number_train' in comparison_stats and 'condition_number_test' in comparison_stats:
            print(f"Train condition number: {comparison_stats['condition_number_train']:.2f}")
            print(f"Test condition number: {comparison_stats['condition_number_test']:.2f}")
        
        print(f"\nResults saved to: {output_dir}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have run Fisher Information analysis on both train and test data for this model.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
