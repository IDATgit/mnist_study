import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def load_eigenvalues(model_name):
    """
    Load eigenvalues for a model from its Fisher analysis directory.
    First tries to load RSVD eigenvalues, falls back to regular eigenvalues if needed.
    
    Args:
        model_name: Name of the model
        
    Returns:
        eigenvalues: Array of eigenvalues for the model
        num_params: Number of parameters in the model
    """
    # Base directory for Fisher analysis outputs
    base_dir = Path(f'model_interpretation/outputs/fisher_analysis/{model_name}')
    
    if not base_dir.exists():
        raise FileNotFoundError(f"No Fisher analysis found for model: {model_name}")
    
    # Try to load RSVD eigenvalues first
    s_path = base_dir / f"{model_name}_S_rsvd.npy"
    
    if s_path.exists():
        print(f"Loading RSVD singular values for {model_name}")
        S = np.load(s_path)
        eigenvalues = S**2  # Convert singular values to eigenvalues
    else:
        # Try regular eigenvalues
        eigen_path = base_dir / f"{model_name}_eigenvalues.npy"
        if eigen_path.exists():
            print(f"Loading regular eigenvalues for {model_name}")
            eigenvalues = np.load(eigen_path)
        else:
            raise FileNotFoundError(f"No eigenvalue files found for model: {model_name}")
    
    # Try to load stats file to get parameter count
    stats_file = base_dir / f"{model_name}_fisher_stats_rsvd.txt"
    if not stats_file.exists():
        stats_file = base_dir / f"{model_name}_fisher_stats.txt"
    
    num_params = None
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            for line in f:
                if 'num_parameters' in line:
                    try:
                        num_params = float(line.split(':')[1].strip())
                    except:
                        pass
    
    return eigenvalues, num_params

def validate_rsvd_approximation(model_name):
    """
    Load full Fisher matrix and RSVD components, compute approximation error.
    
    Args:
        model_name: Name of the model to analyze
    """
    base_dir = Path(f'model_interpretation/outputs/fisher_analysis/{model_name}')
    
    print(f"\nValidating RSVD approximation for {model_name}")
    print("=" * 60)
    
    # Load full Fisher Information Matrix
    fisher_path = base_dir / f"{model_name}_fisher_matrix.npy"
    if not fisher_path.exists():
        raise FileNotFoundError(f"Full Fisher matrix not found: {fisher_path}")
    
    print("Loading full Fisher Information Matrix...")
    F_full = np.load(fisher_path)
    print(f"Full Fisher matrix shape: {F_full.shape}")
    
    # Load RSVD components
    U_path = base_dir / f"{model_name}_U_rsvd.npy"
    S_path = base_dir / f"{model_name}_S_rsvd.npy"
    V_path = base_dir / f"{model_name}_V_rsvd.npy"
    
    if not all(path.exists() for path in [U_path, S_path, V_path]):
        raise FileNotFoundError("RSVD components not found")
    
    print("Loading RSVD components...")
    U = np.load(U_path)
    S = np.load(S_path)
    V = np.load(V_path)
    
    print(f"U shape: {U.shape}")
    print(f"S shape: {S.shape}")
    print(f"V shape: {V.shape}")
    
    # Reconstruct approximated Fisher matrix
    print("\nReconstructing approximated Fisher matrix...")
    
    # For Fisher matrix approximation: F_approx = U @ diag(S) @ V^T
    k = U.shape[1]
    # Reconstruct the approximation
    F_approx = U @ np.diag(S) @ V
    
    print(f"Approximated Fisher matrix shape: {F_approx.shape}")
    
    # Calculate Frobenius norm error
    print("\nCalculating approximation error...")
    error_matrix = F_full - F_approx
    frobenius_error = np.linalg.norm(error_matrix, 'fro')
    frobenius_full = np.linalg.norm(F_full, 'fro')
    
    # Calculate relative error
    relative_error = frobenius_error / frobenius_full
    
    # Print results
    print(f"\nRSVD Approximation Results:")
    print(f"Number of components (k): {k}")
    print(f"Full Fisher matrix Frobenius norm: {frobenius_full:.6e}")
    print(f"Approximated Fisher matrix Frobenius norm: {np.linalg.norm(F_approx, 'fro'):.6e}")
    print(f"Approximation error (Frobenius norm): {frobenius_error:.6e}")
    print(f"Relative error ratio: {relative_error:.6f} ({relative_error*100:.2f}%)")
    
    # Additional statistics
    max_singular_value = S[0] if len(S) > 0 else 0
    min_singular_value = S[-1] if len(S) > 0 else 0
    
    print(f"\nSingular value statistics:")
    print(f"Max singular value: {max_singular_value:.6e}")
    print(f"Min singular value: {min_singular_value:.6e}")
    if min_singular_value > 0:
        print(f"Condition number (max/min): {max_singular_value/min_singular_value:.6e}")
    
    # Compare eigenvalues and create error plots
    print("\nComparing eigenvalues...")
    
    # Load pre-computed eigenvalues from full Fisher analysis
    eigenvals_full_path = base_dir / f"{model_name}_eigenvalues.npy"
    if not eigenvals_full_path.exists():
        raise FileNotFoundError(f"Full eigenvalues not found: {eigenvals_full_path}")
    
    eigenvals_full = np.load(eigenvals_full_path)
    eigenvals_full = np.sort(eigenvals_full)[::-1]  # Sort descending
    
    # Use singular values squared as eigenvalues for RSVD approximation
    eigenvals_approx = S
    eigenvals_approx = np.sort(eigenvals_approx)[::-1]  # Sort descending
    
    # Only compare up to k components (the number of estimated eigenvalues)
    eigenvals_full = eigenvals_full[:k]  # Take only first k eigenvalues from full computation
    eigenvals_approx = eigenvals_approx[:k]  # Take only first k eigenvalues from RSVD
    
    # Calculate eigenvalue errors
    eigenval_absolute_error = np.abs(eigenvals_full - eigenvals_approx)
    eigenval_relative_error = np.abs(eigenval_absolute_error / (np.abs(eigenvals_full) + 1e-12))  # Add small epsilon to avoid division by zero
    
    # Create eigenvalue error plot
    plt.figure(figsize=(12, 8))
    
    # Create subplot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot absolute error on left y-axis (linear scale)
    color = 'tab:red'
    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Absolute Error |λ_full - λ_approx|', color=color)
    line1 = ax1.plot(range(1, len(eigenval_absolute_error) + 1), eigenval_absolute_error, 
                     color=color, linewidth=2, label='Absolute Error')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Set custom y-axis ticks for absolute error (linear)
    abs_min = np.min(eigenval_absolute_error)
    abs_max = np.max(eigenval_absolute_error)
    abs_ticks = np.linspace(abs_min, abs_max, 7)  # 7 linear ticks including min and max
    ax1.set_yticks(abs_ticks)
    ax1.set_yticklabels([f'{tick:.3f}' for tick in abs_ticks])
    
    # Create second y-axis for relative error (linear scale)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Relative Error |Error| / |λ_full|', color=color)
    line2 = ax2.plot(range(1, len(eigenval_relative_error) + 1), eigenval_relative_error, 
                     color=color, linewidth=2, linestyle='--', label='Relative Error')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Set custom y-axis ticks for relative error (linear)
    rel_min = np.min(eigenval_relative_error)
    rel_max = np.max(eigenval_relative_error)
    rel_ticks = np.linspace(rel_min, rel_max, 7)  # 7 linear ticks including min and max
    ax2.set_yticks(rel_ticks)
    ax2.set_yticklabels([f'{tick:.3f}' for tick in rel_ticks])
    
    # Add title and legend
    plt.title(f'Eigenvalue Approximation Errors for {model_name}\n(k={k} components)')
    
    # Combine legends from both axes
    lines = [line1[0], line2[0]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot
    error_plot_path = base_dir / f"{model_name}_eigenvalue_errors.png"
    plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved eigenvalue error plot to: {error_plot_path}")
    plt.close()
    
    # Create a second plot: zoomed-in eigenvalue spectrum comparison for RSVD range only
    plt.figure(figsize=(12, 8))
    
    # Plot only the first k eigenvalues (RSVD range)
    indices = range(1, k + 1)
    plt.semilogy(indices, eigenvals_full[:k], 'b-', linewidth=2, label='Full Computation')
    plt.semilogy(indices, eigenvals_approx[:k], 'r--', linewidth=2, label='RSVD Approximation')
    
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.title(f'Eigenvalue Spectrum Comparison (Zoomed to RSVD Range)\n{model_name} (k={k} components)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the zoomed plot
    zoomed_plot_path = base_dir / f"{model_name}_eigenvalue_spectrum_zoomed.png"
    plt.savefig(zoomed_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved zoomed eigenvalue spectrum plot to: {zoomed_plot_path}")
    plt.close()
    
    # Print some error statistics
    print(f"\nEigenvalue Error Statistics:")
    print(f"Max absolute error: {np.max(eigenval_absolute_error):.6e}")
    print(f"Mean absolute error: {np.mean(eigenval_absolute_error):.6e}")
    print(f"Max relative error: {np.max(eigenval_relative_error):.6e}")
    print(f"Mean relative error: {np.mean(eigenval_relative_error):.6e}")
    
    return {
        'frobenius_error': frobenius_error,
        'frobenius_full': frobenius_full,
        'relative_error': relative_error,
        'num_components': k,
        'max_singular_value': max_singular_value,
        'min_singular_value': min_singular_value,
        'eigenval_absolute_error': eigenval_absolute_error,
        'eigenval_relative_error': eigenval_relative_error
    }

def compare_eigenvalue_distributions(models, save_dir=None):
    """
    Compare eigenvalue distributions between multiple models
    
    Args:
        models: List of model names to compare
        save_dir: Directory to save plots to (optional)
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    all_eigenvalues = []
    all_param_counts = []
    
    # Load eigenvalues for each model
    for model_name in models:
        try:
            eigenvalues, num_params = load_eigenvalues(model_name)
            eigenvalues = np.abs(eigenvalues)
            eigenvalues.sort()
            eigenvalues = eigenvalues[::-1]  # Descending order
            all_eigenvalues.append(eigenvalues)
            all_param_counts.append(num_params)
            print(f"Loaded {len(eigenvalues)} eigenvalues for {model_name}")
        except Exception as e:
            print(f"Error loading eigenvalues for {model_name}: {e}")
    
    if not all_eigenvalues:
        print("No eigenvalues loaded. Exiting.")
        return
    
    # Create eigenvalue spectrum plot
    plt.figure(figsize=(12, 8))
    for i, model_name in enumerate(models):
        if i < len(all_eigenvalues):
            plt.plot(range(1, len(all_eigenvalues[i]) + 1), all_eigenvalues[i], label=f"{model_name}")
    
    plt.title('Eigenvalue Spectrum')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    if save_dir:
        plt.savefig(save_dir / 'eigenvalue_spectrum.png')
        print(f"Saved eigenvalue spectrum to {save_dir / 'eigenvalue_spectrum.png'}")
    
    plt.show()

def print_eigenvalue_stats(models):
    """
    Print statistics about eigenvalues for each model
    
    Args:
        models: List of model names
    """
    print("\n--- Eigenvalue Statistics ---")
    print(f"{'Model':<30} {'Max':<12} {'Min':<12} {'Mean':<12} {'Median':<12} {'Eff. Rank':<12}")
    print("-" * 90)
    
    for model_name in models:
        try:
            eigenvalues, _ = load_eigenvalues(model_name)
            sorted_evals = np.sort(eigenvalues)[::-1]  # Descending order
            
            max_eval = sorted_evals[0]
            min_eval = sorted_evals[-1]
            mean_eval = np.mean(sorted_evals)
            median_eval = np.median(sorted_evals)
            effective_rank = np.sum(sorted_evals) / max_eval
            
            print(f"{model_name:<30} {max_eval:<12.6e} {min_eval:<12.6e} {mean_eval:<12.6e} {median_eval:<12.6e} {effective_rank:<12.2f}")
        except Exception as e:
            print(f"{model_name:<30} Error: {e}")

if __name__ == "__main__":
    # Validate RSVD approximation for small_convnet_10k
    model_name = "small_convnet_10k"
    
    try:
        results = validate_rsvd_approximation(model_name)
        print("\nValidation completed successfully!")
    except Exception as e:
        print(f"Error during validation: {e}")
    
    # Models to compare
    models = ['small_convnet', 'small_convnet_random_images', 'small_convnet_random_labels']
    
    # Compare eigenvalues - both save and show
    # compare_eigenvalue_distributions(models, save_dir='model_interpretation/outputs/eigenvalue_comparison')
    
    # Print statistics
    # print_eigenvalue_stats(models) 