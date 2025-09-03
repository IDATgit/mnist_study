"""
Compare RSVD Eigenvalues Across Multiple Models

This script loads S_rsvd.npy files (singular values from RSVD decomposition) from the
fisher_analysis output directories and creates comparison plots of the eigenvalues
(computed as S^2) across multiple models.

Usage:
    python compare_rsvd_eigenvalues.py --models model1 model2 [--split train/test] [--save-dir path]
    
Example:
    python compare_rsvd_eigenvalues.py --models small_convnet_10k regen_inception_10k
    python compare_rsvd_eigenvalues.py --models small_convnet_10k regen_inception_10k --split test
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import argparse

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


def load_rsvd_eigenvalues(model_name, split='train'):
    """
    Load RSVD singular values (which correspond to Fisher eigenvalues) for a model.
    
    Args:
        model_name: Name of the model
        split: Data split ('train' or 'test')
        
    Returns:
        eigenvalues: Array of eigenvalues (S^2) for the model
        model_info: Dictionary with model metadata
    """
    # Base directory for Fisher analysis outputs
    # Handle both running from model_interpretation/ and from project root
    script_dir = Path(__file__).parent
    base_dir = script_dir / f'outputs/fisher_analysis/{model_name}'
    if not base_dir.exists():
        # Try from project root
        base_dir = Path(f'model_interpretation/outputs/fisher_analysis/{model_name}')
    
    if not base_dir.exists():
        raise FileNotFoundError(f"No Fisher analysis found for model: {model_name}")
    
    # Try to load RSVD singular values
    s_path = base_dir / f"{split}_{model_name}_S_rsvd.npy"
    
    if not s_path.exists():
        raise FileNotFoundError(f"No RSVD singular values found for model: {model_name} at {s_path}")
    
    print(f"Loading RSVD singular values for {model_name} ({split} split)")
    S = np.load(s_path)
    
    eigenvalues = S
    
    # Sort eigenvalues in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Try to load stats file to get additional information
    stats_file = base_dir / f"{split}_{model_name}_fisher_stats_rsvd.txt"
    model_info = {'num_params': None, 'num_components': len(eigenvalues)}
    
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            content = f.read()
            # Extract number of parameters
            for line in content.split('\n'):
                if 'num_parameters' in line.lower():
                    try:
                        model_info['num_params'] = float(line.split(':')[1].strip())
                    except:
                        pass
    
    return eigenvalues, model_info


def compare_rsvd_eigenvalues(models, split='train', save_path=None):
    """
    Compare RSVD eigenvalues between multiple models - simple plot only.
    """
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Load and plot each model
    for i, model_name in enumerate(models):
        try:
            eigenvalues, _ = load_rsvd_eigenvalues(model_name, split)
            indices = range(1, len(eigenvalues) + 1)
            plt.semilogy(indices, eigenvalues, 
                        linewidth=2, label=model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue (log scale)')
    plt.title('RSVD Eigenvalue Spectrum Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Generate default save path if not provided
    if save_path is None:
        # Create rsvd_comparison directory in fisher_analysis
        script_dir = Path(__file__).parent
        comparison_dir = script_dir / 'outputs/fisher_analysis/rsvd_comparison'
        if not comparison_dir.exists():
            # Try from project root
            comparison_dir = Path('model_interpretation/outputs/fisher_analysis/rsvd_comparison')
        
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with all model names
        model_names_str = '_vs_'.join(models)
        filename = f"{model_names_str}_{split}_comparison.png"
        save_path = comparison_dir / filename
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")
    
    plt.close()


def main():
    """
    Main function to run the RSVD eigenvalue comparison.
    """
    parser = argparse.ArgumentParser(description='Compare RSVD eigenvalues across multiple models')
    parser.add_argument('--models', nargs='+', 
                        default=['regen_inception_10k', 'regen_inception_10k_random_labels'],
                        help='List of model names to compare (default: regen_inception_10k variants)')
    parser.add_argument('--split', choices=['train', 'test'], default='train',
                        help='Data split to use (default: train)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save plot (default: auto-generated in fisher_analysis/rsvd_comparison/)')
    
    args = parser.parse_args()
    
    # Run the comparison
    compare_rsvd_eigenvalues(
        models=args.models,
        split=args.split,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()
