import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    model_name = "small_convnet_10k"
    base_dir = Path(f'model_interpretation/outputs/fisher_analysis/{model_name}')
    
    # Load full computation eigenvalues
    full_eigenvalues = np.load(base_dir / f"{model_name}_eigenvalues.npy")
    
    # Load RSVD eigenvalues (from singular values)
    S = np.load(base_dir / f"{model_name}_S_rsvd.npy")
    rsvd_eigenvalues = S  # Convert singular values to eigenvalues
    
    # Sort eigenvalues in descending order
    full_sorted = np.sort(np.abs(full_eigenvalues))[::-1]
    rsvd_sorted = np.sort(np.abs(rsvd_eigenvalues))[::-1]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(full_sorted) + 1), full_sorted, 'b-', label='Full Computation', alpha=0.7)
    plt.plot(range(1, len(rsvd_sorted) + 1), rsvd_sorted, 'r-', label='RSVD', alpha=0.7)
    
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    plt.title(f'Eigenvalue Spectrum: {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save in the same directory as the eigenvalue files
    output_path = base_dir / f'{model_name}_eigenvalue_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()