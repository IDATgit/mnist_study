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
    # Models to compare
    models = ['small_convnet', 'small_convnet_random_images', 'small_convnet_random_labels']
    
    # Compare eigenvalues - both save and show
    compare_eigenvalue_distributions(models, save_dir='model_interpretation/outputs/eigenvalue_comparison')
    
    # Print statistics
    print_eigenvalue_stats(models) 