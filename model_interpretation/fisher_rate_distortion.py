import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_fisher_rate_distortion(model_name="small_convnet_10k"):
    """
    Analyze rate-distortion tradeoff for Fisher information computation.
    Rate: Fraction of samples included
    Distortion: Loss of Fisher information power
    """
    base_dir = Path(f'outputs/fisher_analysis/{model_name}')
    
    # Load the diagnostic vectors
    print(f"Loading diagnostic vectors for {model_name}...")
    prob_values = np.load(base_dir / f"{model_name}_prob_values.npy")
    fisher_matrix_norms = np.load(base_dir / f"{model_name}_fisher_matrix_norms.npy")
    
    print(f"Loaded {len(prob_values)} data points")
    
    # Calculate total Fisher "power" (sum of squared norms)
    fisher_power = fisher_matrix_norms**2
    total_power = np.sum(fisher_power)
    print(f"Total Fisher power: {total_power:.6e}")
    
    # Filter out zero Fisher norms
    valid_mask = fisher_matrix_norms > 0
    prob_valid = prob_values[valid_mask]
    fisher_power_valid = fisher_power[valid_mask]
    
    print(f"Valid samples: {len(prob_valid)} (excluding {np.sum(~valid_mask)} zero Fisher norms)")
    
    # Define threshold ranges to test
    # x ranges from 0 to 0.5 (we skip probs < x and > 1-x)
    x_thresholds = np.logspace(-8, -1, 50)  # From 1e-8 to 0.1
    
    rates = []  # Fraction of samples included
    powers_captured = []  # Fraction of total power captured
    distortions = []  # Fraction of power lost
    
    for x in x_thresholds:
        # Include samples with probability in [x, 1-x]
        include_mask = (prob_valid >= x) & (prob_valid <= (1 - x))
        
        # Calculate rate (fraction of samples included)
        rate = np.sum(include_mask) / len(prob_valid)
        
        # Calculate power captured
        power_captured = np.sum(fisher_power_valid[include_mask]) / total_power
        
        # Calculate distortion (power lost)
        distortion = 1 - power_captured
        
        rates.append(rate)
        powers_captured.append(power_captured)
        distortions.append(distortion)
        
        if x in [1e-6, 1e-4, 1e-2, 0.1]:
            print(f"Threshold x={x:.1e}: Rate={rate:.3f}, Power captured={power_captured:.3f}, Samples excluded: p<{x:.1e} or p>{1-x:.1e}")
    
    rates = np.array(rates)
    powers_captured = np.array(powers_captured)
    distortions = np.array(distortions)
    
    # Create rate-distortion plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Rate vs Power Captured
    ax1.plot(rates, powers_captured, 'b.-', linewidth=2, markersize=4)
    ax1.set_xlabel('Rate (Fraction of Samples Included)')
    ax1.set_ylabel('Power Captured (Fraction of Total)')
    ax1.set_title('Rate vs Fisher Power Captured')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Add diagonal reference line
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect efficiency')
    ax1.legend()
    
    # Plot 2: Rate vs Distortion (log scale)
    ax2.semilogy(rates, distortions, 'r.-', linewidth=2, markersize=4)
    ax2.set_xlabel('Rate (Fraction of Samples Included)')
    ax2.set_ylabel('Distortion (Fraction of Power Lost)')
    ax2.set_title('Rate-Distortion Curve (Log Scale)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    # Plot 3: Threshold vs Rate and Power
    ax3.loglog(x_thresholds, rates, 'b.-', label='Sample Rate', linewidth=2, markersize=4)
    ax3.loglog(x_thresholds, powers_captured, 'g.-', label='Power Captured', linewidth=2, markersize=4)
    ax3.set_xlabel('Threshold x (exclude p < x and p > 1-x)')
    ax3.set_ylabel('Fraction')
    ax3.set_title('Threshold vs Rate and Power Captured')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.invert_xaxis()  # Smaller thresholds (more inclusive) on the right
    
    # Plot 4: Efficiency (Power/Rate)
    efficiency = powers_captured / (rates + 1e-10)  # Avoid division by zero
    ax4.semilogx(x_thresholds, efficiency, 'm.-', linewidth=2, markersize=4)
    ax4.set_xlabel('Threshold x')
    ax4.set_ylabel('Efficiency (Power Captured / Sample Rate)')
    ax4.set_title('Fisher Information Efficiency vs Threshold')
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()
    ax4.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Perfect efficiency')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_path = base_dir / f'{model_name}_fisher_rate_distortion.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Rate-distortion analysis saved to: {output_path}")
    
    # Save numerical results
    results = {
        'x_thresholds': x_thresholds,
        'rates': rates,
        'powers_captured': powers_captured,
        'distortions': distortions,
        'efficiency': efficiency
    }
    
    results_path = base_dir / f'{model_name}_rate_distortion_data.npz'
    np.savez(results_path, **results)
    print(f"Numerical results saved to: {results_path}")
    
    # Print key insights
    print("\n=== KEY INSIGHTS ===")
    
    # Find optimal operating points
    # 90% power with minimum samples
    idx_90 = np.argmax(powers_captured >= 0.9)
    if powers_captured[idx_90] >= 0.9:
        print(f"To capture 90% of Fisher power: include {rates[idx_90]:.1%} of samples (threshold x={x_thresholds[idx_90]:.2e})")
    
    # 95% power
    idx_95 = np.argmax(powers_captured >= 0.95)
    if powers_captured[idx_95] >= 0.95:
        print(f"To capture 95% of Fisher power: include {rates[idx_95]:.1%} of samples (threshold x={x_thresholds[idx_95]:.2e})")
    
    # 99% power
    idx_99 = np.argmax(powers_captured >= 0.99)
    if powers_captured[idx_99] >= 0.99:
        print(f"To capture 99% of Fisher power: include {rates[idx_99]:.1%} of samples (threshold x={x_thresholds[idx_99]:.2e})")
    
    # Maximum efficiency point
    max_eff_idx = np.argmax(efficiency)
    print(f"Maximum efficiency: {efficiency[max_eff_idx]:.2f} at threshold x={x_thresholds[max_eff_idx]:.2e}")
    print(f"  (captures {powers_captured[max_eff_idx]:.1%} power with {rates[max_eff_idx]:.1%} samples)")
    
    return results

def main():
    analyze_fisher_rate_distortion("small_convnet_10k")

if __name__ == "__main__":
    main()