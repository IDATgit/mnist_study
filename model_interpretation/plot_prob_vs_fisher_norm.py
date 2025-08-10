import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def plot_probability_vs_fisher_norm(model_name="small_convnet_10k"):
    """
    Plot probability vs Fisher matrix L2 norm scatter plot.
    X-axis: Probability values
    Y-axis: Fisher matrix L2 norms (log scale)
    """
    base_dir = Path(f'outputs/fisher_analysis/{model_name}')
    
    # Load the diagnostic vectors
    print(f"Loading diagnostic vectors for {model_name}...")
    prob_values = np.load(base_dir / f"{model_name}_prob_values.npy")
    fisher_matrix_norms = np.load(base_dir / f"{model_name}_fisher_matrix_norms.npy")
    
    print(f"Loaded {len(prob_values)} data points")
    print(f"Probability range: [{prob_values.min():.6f}, {prob_values.max():.6f}]")
    print(f"Fisher norm range: [{fisher_matrix_norms.min():.6e}, {fisher_matrix_norms.max():.6e}]")
    
    # Split data into low and high probability regions
    valid_fisher_mask = fisher_matrix_norms > 0
    
    # Low probabilities (p <= 0.5): plot p directly
    low_prob_mask = valid_fisher_mask & (prob_values <= 0.5) & (prob_values > 1e-10)
    prob_low = prob_values[low_prob_mask]
    fisher_low = fisher_matrix_norms[low_prob_mask]
    
    # High probabilities (p > 0.5): plot 1-p
    high_prob_mask = valid_fisher_mask & (prob_values > 0.5) & (prob_values < (1 - 1e-10))
    prob_high_original = prob_values[high_prob_mask]
    prob_high = 1 - prob_high_original  # Plot 1-p for high probabilities
    fisher_high = fisher_matrix_norms[high_prob_mask]
    
    print(f"Low probabilities (p <= 0.5): {len(prob_low)} data points")
    print(f"High probabilities (p > 0.5): {len(prob_high)} data points, plotted as 1-p")
    print(f"Low prob range: [{prob_low.min():.6e}, {prob_low.max():.6f}]")
    print(f"High prob range (1-p): [{prob_high.min():.6e}, {prob_high.max():.6f}]")
    
    # Create the scatter plot with log scale on both axes
    plt.figure(figsize=(12, 8))
    
    # Plot low probabilities in blue
    plt.scatter(prob_low, fisher_low, alpha=0.6, s=1, c='blue', label=f'Low prob (p ≤ 0.5, n={len(prob_low)})')
    
    # Plot high probabilities (as 1-p) in red
    plt.scatter(prob_high, fisher_high, alpha=0.6, s=1, c='red', label=f'High prob (1-p, n={len(prob_high)})')
    
    plt.xlabel('Probability / (1 - Probability) (log scale)')
    plt.ylabel('Fisher Matrix L2 Norm (log scale)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Probability vs Fisher Matrix L2 Norm (Log-Log Scale)\n{model_name}\nBlue: p ≤ 0.5, Red: 1-p for p > 0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = base_dir / f'{model_name}_prob_vs_fisher_norm.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")

def main():
    plot_probability_vs_fisher_norm("small_convnet_10k")

if __name__ == "__main__":
    main()