import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
import argparse

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


def load_eigenvector_data(model_name, data_type):
    """
    Load eigenvector data (U matrix) from RSVD Fisher analysis.
    
    Args:
        model_name: Name of the model
        data_type: 'train' or 'test'
        
    Returns:
        U: Eigenvectors matrix (num_params x k)
        S: Eigenvalues (singular values)
        stats: Statistics dictionary
    """
    # Base directory for Fisher analysis outputs
    base_dir = Path(f'model_interpretation/outputs/fisher_analysis/{model_name}')
    
    if not base_dir.exists():
        raise FileNotFoundError(f"No Fisher analysis found for model: {model_name}")
    
    # Load eigenvectors (U matrix)
    u_path = base_dir / f"{data_type}_{model_name}_U_rsvd.npy"
    if not u_path.exists():
        raise FileNotFoundError(f"No {data_type} eigenvectors found for model: {model_name}")
    
    print(f"Loading {data_type} eigenvectors for {model_name}")
    U = np.load(u_path)
    
    # Load eigenvalues (S matrix)
    s_path = base_dir / f"{data_type}_{model_name}_S_rsvd.npy"
    if not s_path.exists():
        raise FileNotFoundError(f"No {data_type} eigenvalues found for model: {model_name}")
    
    S = np.load(s_path)
    
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
    
    print(f"Loaded eigenvectors shape: {U.shape}")
    print(f"Loaded eigenvalues shape: {S.shape}")
    
    return U, S, stats


def quantify_bow_tie_pattern(eigenvector, num_params):
    """
    Quantify the bow-tie pattern in an eigenvector.
    
    Args:
        eigenvector: The eigenvector to analyze
        num_params: Total number of parameters
        
    Returns:
        Dictionary with quantitative measures
    """
    abs_values = np.abs(eigenvector)
    
    # Define regions (adjust these based on typical network architectures)
    early_cutoff = int(0.1 * num_params)  # First 10% of parameters
    late_cutoff = int(0.9 * num_params)   # Last 10% of parameters
    
    early_region = abs_values[:early_cutoff]
    middle_region = abs_values[early_cutoff:late_cutoff]
    late_region = abs_values[late_cutoff:]
    
    # Calculate importance scores
    early_importance = np.mean(early_region)
    middle_importance = np.mean(middle_region)
    late_importance = np.mean(late_region)
    
    # Calculate concentration measures
    total_mass = np.sum(abs_values)
    early_mass_fraction = np.sum(early_region) / total_mass
    middle_mass_fraction = np.sum(middle_region) / total_mass
    late_mass_fraction = np.sum(late_region) / total_mass
    
    # Bow-tie score: higher when early and late are important relative to middle
    bow_tie_score = (early_importance + late_importance) / (2 * middle_importance) if middle_importance > 0 else float('inf')
    
    # Sparsity measure
    sparsity = np.sum(abs_values < 0.01 * np.max(abs_values)) / num_params
    
    return {
        'early_importance': early_importance,
        'middle_importance': middle_importance,
        'late_importance': late_importance,
        'early_mass_fraction': early_mass_fraction,
        'middle_mass_fraction': middle_mass_fraction,
        'late_mass_fraction': late_mass_fraction,
        'bow_tie_score': bow_tie_score,
        'sparsity': sparsity,
        'max_value': np.max(abs_values),
        'mean_value': np.mean(abs_values),
        'std_value': np.std(eigenvector)
    }


def plot_first_eigenvector(U, S, model_name, data_type, output_dir):
    """
    Plot the first eigenvector (highest eigenvalue) against parameter index.
    
    Args:
        U: Eigenvectors matrix (num_params x k)
        S: Eigenvalues
        model_name: Name of the model
        data_type: 'train' or 'test'
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the first eigenvector (corresponding to largest eigenvalue)
    first_eigenvector = U[:, 0]
    first_eigenvalue = S[0]
    num_params = len(first_eigenvector)
    
    # Quantify the bow-tie pattern
    pattern_metrics = quantify_bow_tie_pattern(first_eigenvector, num_params)
    
    # Create parameter indices
    param_indices = np.arange(1, num_params + 1)
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Plot the eigenvector values
    plt.plot(param_indices, first_eigenvector, 'b-', linewidth=1, alpha=0.8)
    
    plt.title(f'First Eigenvector - {data_type.title()} Data\n{model_name} (Eigenvalue: {first_eigenvalue:.6f})')
    plt.xlabel('Parameter Index')
    plt.ylabel('Eigenvector Component Value')
    plt.grid(True, alpha=0.3)
    
    # Add comprehensive statistics to the plot
    textstr = f'''Max |value|: {pattern_metrics['max_value']:.6f}
Mean |value|: {pattern_metrics['mean_value']:.6f}
Std: {pattern_metrics['std_value']:.6f}
Bow-tie Score: {pattern_metrics['bow_tie_score']:.3f}
Early/Middle/Late: {pattern_metrics['early_importance']:.4f}/{pattern_metrics['middle_importance']:.4f}/{pattern_metrics['late_importance']:.4f}
Sparsity: {pattern_metrics['sparsity']:.3f}'''
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{data_type}_{model_name}_first_eigenvector.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a zoomed-in version showing more detail
    plt.figure(figsize=(15, 8))
    
    # Plot with markers for better visibility
    plt.plot(param_indices, first_eigenvector, 'b-', linewidth=0.5, alpha=0.8)
    plt.scatter(param_indices[::max(1, num_params//1000)], 
               first_eigenvector[::max(1, num_params//1000)], 
               c='red', s=10, alpha=0.6, zorder=5)
    
    plt.title(f'First Eigenvector (with sample points) - {data_type.title()} Data\n{model_name} (Eigenvalue: {first_eigenvalue:.6f})')
    plt.xlabel('Parameter Index')
    plt.ylabel('Eigenvector Component Value')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{data_type}_{model_name}_first_eigenvector_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save quantitative analysis
    analysis_file = output_dir / f'{data_type}_{model_name}_eigenvector_analysis.txt'
    with open(analysis_file, 'w') as f:
        f.write(f"First Eigenvector Analysis for {model_name} ({data_type} data)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Eigenvalue: {first_eigenvalue:.6f}\n")
        f.write(f"Number of parameters: {num_params}\n\n")
        
        f.write("BOW-TIE PATTERN ANALYSIS:\n")
        f.write(f"  Bow-tie Score: {pattern_metrics['bow_tie_score']:.6f}\n")
        f.write(f"  Early Region Importance (first 10%): {pattern_metrics['early_importance']:.6f}\n")
        f.write(f"  Middle Region Importance (middle 80%): {pattern_metrics['middle_importance']:.6f}\n")
        f.write(f"  Late Region Importance (last 10%): {pattern_metrics['late_importance']:.6f}\n\n")
        
        f.write("MASS DISTRIBUTION:\n")
        f.write(f"  Early Region Mass Fraction: {pattern_metrics['early_mass_fraction']:.4f}\n")
        f.write(f"  Middle Region Mass Fraction: {pattern_metrics['middle_mass_fraction']:.4f}\n")
        f.write(f"  Late Region Mass Fraction: {pattern_metrics['late_mass_fraction']:.4f}\n\n")
        
        f.write("GENERAL STATISTICS:\n")
        f.write(f"  Max absolute value: {pattern_metrics['max_value']:.6f}\n")
        f.write(f"  Mean absolute value: {pattern_metrics['mean_value']:.6f}\n")
        f.write(f"  Standard deviation: {pattern_metrics['std_value']:.6f}\n")
        f.write(f"  Sparsity (% near zero): {pattern_metrics['sparsity']:.4f}\n")
    
    print(f"First eigenvector bow-tie analysis:")
    print(f"  Eigenvalue: {first_eigenvalue:.6f}")
    print(f"  Bow-tie Score: {pattern_metrics['bow_tie_score']:.3f}")
    print(f"  Early/Middle/Late importance: {pattern_metrics['early_importance']:.4f}/{pattern_metrics['middle_importance']:.4f}/{pattern_metrics['late_importance']:.4f}")
    print(f"  Early/Middle/Late mass: {pattern_metrics['early_mass_fraction']:.3f}/{pattern_metrics['middle_mass_fraction']:.3f}/{pattern_metrics['late_mass_fraction']:.3f}")
    print(f"  Sparsity: {pattern_metrics['sparsity']:.3f}")
    print(f"  Number of parameters: {num_params}")
    
    return pattern_metrics


def plot_multiple_eigenvectors(U, S, model_name, data_type, output_dir, num_vectors=5):
    """
    Plot multiple eigenvectors for comparison.
    
    Args:
        U: Eigenvectors matrix (num_params x k)
        S: Eigenvalues
        model_name: Name of the model
        data_type: 'train' or 'test'
        output_dir: Directory to save plots
        num_vectors: Number of top eigenvectors to plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_params = U.shape[0]
    num_vectors = min(num_vectors, U.shape[1])
    param_indices = np.arange(1, num_params + 1)
    
    # Create subplot for multiple eigenvectors
    fig, axes = plt.subplots(num_vectors, 1, figsize=(15, 3*num_vectors), sharex=True)
    if num_vectors == 1:
        axes = [axes]
    
    for i in range(num_vectors):
        eigenvector = U[:, i]
        eigenvalue = S[i]
        
        axes[i].plot(param_indices, eigenvector, 'b-', linewidth=1, alpha=0.8)
        axes[i].set_title(f'Eigenvector {i+1} (Eigenvalue: {eigenvalue:.6f})')
        axes[i].set_ylabel('Component Value')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        max_val = np.max(np.abs(eigenvector))
        mean_val = np.mean(np.abs(eigenvector))
        textstr = f'Max |val|: {max_val:.4f}\nMean |val|: {mean_val:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes[i].text(0.02, 0.98, textstr, transform=axes[i].transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)
    
    axes[-1].set_xlabel('Parameter Index')
    plt.suptitle(f'Top {num_vectors} Eigenvectors - {data_type.title()} Data\n{model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'{data_type}_{model_name}_top_{num_vectors}_eigenvectors.png', dpi=300, bbox_inches='tight')
    plt.close()


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
            # Check if the directory has both train and test RSVD files
            train_u = model_dir / f"train_{model_dir.name}_U_rsvd.npy"
            test_u = model_dir / f"test_{model_dir.name}_U_rsvd.npy"
            
            if train_u.exists() or test_u.exists():
                available_models.append(model_dir.name)
    
    return sorted(available_models)


def main():
    """
    Main function to visualize eigenvectors from Fisher Information analysis.
    """
    parser = argparse.ArgumentParser(description='Visualize Fisher Information Matrix eigenvectors')
    parser.add_argument('--model', type=str, help='Model name to analyze')
    parser.add_argument('--data-type', type=str, choices=['train', 'test'], default='train', help='Data type to analyze (train or test)')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    
    args = parser.parse_args()
    
    # Get available models that have Fisher analysis data
    available_models = get_available_fisher_models()
    
    if not available_models:
        print("No models with Fisher analysis data found!")
        print("Please run Fisher Information analysis first using fisher_information_rsvd.py")
        return
    
    # List available models if requested
    if args.list_models:
        print("Available models for eigenvector visualization:")
        for i, model_name in enumerate(available_models):
            print(f"  {i}: {model_name}")
        return
    
    # If no model specified, list available and exit
    if not args.model:
        print("Available models for eigenvector visualization:")
        for i, model_name in enumerate(available_models):
            print(f"  {i}: {model_name}")
        print(f"\nUsage: python {sys.argv[0]} --model MODEL_NAME [--data-type train|test]")
        print(f"   or: python {sys.argv[0]} --list-models")
        return
    
    selected_model = args.model
    selected_data_type = args.data_type
    
    # Check if the specified model exists
    if selected_model not in available_models:
        print(f"Model '{selected_model}' not found!")
        print("Available models:", ', '.join(available_models))
        return
    
    # Check what data types are available for this model
    base_dir = Path(f'model_interpretation/outputs/fisher_analysis/{selected_model}')
    train_available = (base_dir / f"train_{selected_model}_U_rsvd.npy").exists()
    test_available = (base_dir / f"test_{selected_model}_U_rsvd.npy").exists()
    
    # Check if requested data type is available
    if selected_data_type == 'train' and not train_available:
        print(f"Train data not available for model '{selected_model}'")
        if test_available:
            print("Test data is available. Use --data-type test")
        return
    elif selected_data_type == 'test' and not test_available:
        print(f"Test data not available for model '{selected_model}'")
        if train_available:
            print("Train data is available. Use --data-type train")
        return
    
    print(f"Analyzing eigenvectors for: {selected_model} ({selected_data_type} data)")
    
    try:
        # Load eigenvector data
        U, S, stats = load_eigenvector_data(selected_model, selected_data_type)
        
        # Create output directory
        output_dir = Path(f'model_interpretation/outputs/fisher_analysis/{selected_model}/eigenvector_analysis')
        
        # Plot first eigenvector
        print("Plotting first eigenvector...")
        pattern_metrics = plot_first_eigenvector(U, S, selected_model, selected_data_type, output_dir)
        
        # Plot multiple eigenvectors
        print("Plotting top 5 eigenvectors...")
        plot_multiple_eigenvectors(U, S, selected_model, selected_data_type, output_dir, num_vectors=5)
        
        # If both train and test data exist, compare them
        other_data_type = 'test' if selected_data_type == 'train' else 'train'
        other_data_available = (base_dir / f"{other_data_type}_{selected_model}_U_rsvd.npy").exists()
        
        if other_data_available:
            print(f"\nComparing with {other_data_type} data...")
            try:
                U_other, S_other, stats_other = load_eigenvector_data(selected_model, other_data_type)
                pattern_metrics_other = plot_first_eigenvector(U_other, S_other, selected_model, other_data_type, output_dir)
                
                # Compare the patterns
                print(f"\nBOW-TIE PATTERN COMPARISON:")
                print(f"  {selected_data_type.title()} bow-tie score: {pattern_metrics['bow_tie_score']:.3f}")
                print(f"  {other_data_type.title()} bow-tie score: {pattern_metrics_other['bow_tie_score']:.3f}")
                print(f"  Pattern consistency: {'HIGH' if abs(pattern_metrics['bow_tie_score'] - pattern_metrics_other['bow_tie_score']) < 1.0 else 'LOW'}")
                
                # Save comparison
                comparison_file = output_dir / f'{selected_model}_train_test_pattern_comparison.txt'
                with open(comparison_file, 'w') as f:
                    f.write(f"Bow-tie Pattern Comparison for {selected_model}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Train bow-tie score: {pattern_metrics.get('bow_tie_score', 'N/A') if selected_data_type == 'train' else pattern_metrics_other.get('bow_tie_score', 'N/A'):.6f}\n")
                    f.write(f"Test bow-tie score: {pattern_metrics_other.get('bow_tie_score', 'N/A') if selected_data_type == 'train' else pattern_metrics.get('bow_tie_score', 'N/A'):.6f}\n")
                    f.write(f"Score difference: {abs(pattern_metrics['bow_tie_score'] - pattern_metrics_other['bow_tie_score']):.6f}\n")
                
            except Exception as e:
                print(f"Could not analyze {other_data_type} data: {e}")
        
        print(f"\nEigenvector visualizations saved to: {output_dir}")
        
        # Summary for research confidence
        print(f"\nRESEARCH CONFIDENCE INDICATORS:")
        print(f"  Bow-tie score: {pattern_metrics['bow_tie_score']:.3f} (>2.0 suggests strong pattern)")
        print(f"  Early region dominance: {pattern_metrics['early_mass_fraction']:.3f} (>0.3 suggests early importance)")
        print(f"  Late region dominance: {pattern_metrics['late_mass_fraction']:.3f} (>0.1 suggests late importance)")
        print(f"  Sparsity: {pattern_metrics['sparsity']:.3f} (>0.8 suggests sparse structure)")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
