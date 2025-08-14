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
    
    # Add some statistics to the plot
    max_val = np.max(np.abs(first_eigenvector))
    mean_val = np.mean(np.abs(first_eigenvector))
    std_val = np.std(first_eigenvector)
    
    textstr = f'Max |value|: {max_val:.6f}\nMean |value|: {mean_val:.6f}\nStd: {std_val:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
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
    
    print(f"First eigenvector statistics:")
    print(f"  Eigenvalue: {first_eigenvalue:.6f}")
    print(f"  Max absolute value: {max_val:.6f}")
    print(f"  Mean absolute value: {mean_val:.6f}")
    print(f"  Standard deviation: {std_val:.6f}")
    print(f"  Number of parameters: {num_params}")


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
    # Get available models that have Fisher analysis data
    available_models = get_available_fisher_models()
    
    if not available_models:
        print("No models with Fisher analysis data found!")
        print("Please run Fisher Information analysis first using fisher_information_rsvd.py")
        return
    
    # Display available models
    print("Available models for eigenvector visualization:")
    for i, model_name in enumerate(available_models, 1):
        print(f"{i}. {model_name}")
    
    # Get user choice for model
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
    
    # Check what data types are available for this model
    base_dir = Path(f'model_interpretation/outputs/fisher_analysis/{selected_model}')
    train_available = (base_dir / f"train_{selected_model}_U_rsvd.npy").exists()
    test_available = (base_dir / f"test_{selected_model}_U_rsvd.npy").exists()
    
    print(f"\nSelected model: {selected_model}")
    print("Available data types:")
    available_data_types = []
    if train_available:
        print("1. Train data")
        available_data_types.append('train')
    if test_available:
        print("2. Test data")
        available_data_types.append('test')
    
    if not available_data_types:
        print("No eigenvector data found for this model!")
        return
    
    # Get user choice for data type
    if len(available_data_types) == 1:
        selected_data_type = available_data_types[0]
        print(f"Using {selected_data_type} data (only option available)")
    else:
        while True:
            try:
                data_choice = int(input(f"\nSelect data type (1-{len(available_data_types)}): ")) - 1
                if 0 <= data_choice < len(available_data_types):
                    selected_data_type = available_data_types[data_choice]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(available_data_types)}")
            except ValueError:
                print("Please enter a valid number")
    
    print(f"\nAnalyzing eigenvectors for: {selected_model} ({selected_data_type} data)")
    
    try:
        # Load eigenvector data
        U, S, stats = load_eigenvector_data(selected_model, selected_data_type)
        
        # Create output directory
        output_dir = Path(f'model_interpretation/outputs/fisher_analysis/{selected_model}/eigenvector_analysis')
        
        # Plot first eigenvector
        print("Plotting first eigenvector...")
        plot_first_eigenvector(U, S, selected_model, selected_data_type, output_dir)
        
        # Plot multiple eigenvectors
        print("Plotting top 5 eigenvectors...")
        plot_multiple_eigenvectors(U, S, selected_model, selected_data_type, output_dir, num_vectors=5)
        
        print(f"\nEigenvector visualizations saved to: {output_dir}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
