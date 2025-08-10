import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    model_name = "small_convnet_random_labels_10k"
    base_dir = Path(f'model_interpretation/outputs/fisher_analysis/{model_name}')
    confident_dir = Path(f'model_interpretation/outputs/fisher_analysis/{model_name}_skip_confident')
    
    # Load full computation eigenvalues
    full_eigenvalues = np.load(base_dir / f"{model_name}_eigenvalues.npy")
    
    # Load confident filtering eigenvalues
    confident_eigenvalues = np.load(confident_dir / f"{model_name}_skip_confident_eigenvalues.npy")
    
    # Sort eigenvalues in descending order
    full_sorted = np.sort(np.abs(full_eigenvalues))[::-1]
    confident_sorted = np.sort(np.abs(confident_eigenvalues))[::-1]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(full_sorted) + 1), full_sorted, 'b-', label='Full Computation', alpha=0.7)
    plt.plot(range(1, len(confident_sorted) + 1), confident_sorted, 'r-', label='Skip Confident', alpha=0.7)
    
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    plt.title(f'Eigenvalue Spectrum: {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save in the base directory
    output_path = base_dir / f'{model_name}_eigenvalue_comparison_full_vs_confident.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()