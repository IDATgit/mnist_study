import torch
import torch.nn as nn
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

# Import our existing analysis functions
from visualize_eigenvectors import quantify_bow_tie_pattern
from fisher_information_rsvd import calculate_fisher_rsvd, analyze_fisher_information_rsvd


def create_random_network(trained_model):
    """
    Create a random network with the same architecture as the trained model.
    
    Args:
        trained_model: The trained model to copy architecture from
        
    Returns:
        Random network with same architecture but random weights
    """
    # Get the model class and create a new instance
    model_class = trained_model.__class__
    
    # Create new model with same architecture
    random_model = model_class()
    
    # Ensure it's in the same state (eval/train mode)
    random_model.train()
    
    # Move to same device as trained model
    device = next(trained_model.parameters()).device
    random_model = random_model.to(device)
    
    print(f"Created random network: {model_class.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in random_model.parameters())}")
    
    return random_model


def compare_trained_vs_random(trained_model, data_loader, model_name, output_dir):
    """
    Compare Fisher Information eigenvector patterns between trained and random networks.
    
    Args:
        trained_model: The trained model
        data_loader: Data loader for Fisher computation
        model_name: Name of the model
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = next(trained_model.parameters()).device
    
    # Create random network with same architecture
    print("Creating random network...")
    random_model = create_random_network(trained_model)
    
    # Compute Fisher Information for both networks
    print("\nComputing Fisher Information for random network...")
    k = 1000  # Number of components
    power_iterations = 1
    
    U_random, S_random, V_random, error_random = calculate_fisher_rsvd(
        random_model, data_loader, k, power_iterations
    )
    
    print("Random network Fisher computation completed.")
    
    # Load trained network Fisher data if available
    fisher_dir = Path(f'model_interpretation/outputs/fisher_analysis/{model_name}')
    train_u_file = fisher_dir / f"train_{model_name}_U_rsvd.npy"
    train_s_file = fisher_dir / f"train_{model_name}_S_rsvd.npy"
    
    if train_u_file.exists() and train_s_file.exists():
        print("Loading trained network Fisher data...")
        U_trained = np.load(train_u_file)
        S_trained = np.load(train_s_file)
    else:
        print("No trained Fisher data found. Computing for trained network...")
        U_trained, S_trained, V_trained, error_trained = calculate_fisher_rsvd(
            trained_model, data_loader, k, power_iterations
        )
    
    # Analyze patterns for both networks
    print("\nAnalyzing eigenvector patterns...")
    
    # Get first eigenvectors
    first_eigenvector_random = U_random[:, 0]
    first_eigenvector_trained = U_trained[:, 0]
    
    num_params = len(first_eigenvector_random)
    
    # Quantify patterns
    pattern_random = quantify_bow_tie_pattern(first_eigenvector_random, num_params)
    pattern_trained = quantify_bow_tie_pattern(first_eigenvector_trained, num_params)
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    param_indices = np.arange(1, num_params + 1)
    
    # Plot random network
    ax1.plot(param_indices, first_eigenvector_random, 'r-', linewidth=1, alpha=0.8)
    ax1.set_title(f'Random Network First Eigenvector\n{model_name} (Eigenvalue: {S_random[0]:.6f})')
    ax1.set_ylabel('Eigenvector Component Value')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics for random
    textstr_random = f'''Bow-tie Score: {pattern_random['bow_tie_score']:.3f}
Early/Middle/Late: {pattern_random['early_importance']:.4f}/{pattern_random['middle_importance']:.4f}/{pattern_random['late_importance']:.4f}
Early/Middle/Late Mass: {pattern_random['early_mass_fraction']:.3f}/{pattern_random['middle_mass_fraction']:.3f}/{pattern_random['late_mass_fraction']:.3f}
Sparsity: {pattern_random['sparsity']:.3f}'''
    props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.8)
    ax1.text(0.02, 0.98, textstr_random, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    # Plot trained network
    ax2.plot(param_indices, first_eigenvector_trained, 'b-', linewidth=1, alpha=0.8)
    ax2.set_title(f'Trained Network First Eigenvector\n{model_name} (Eigenvalue: {S_trained[0]:.6f})')
    ax2.set_xlabel('Parameter Index')
    ax2.set_ylabel('Eigenvector Component Value')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics for trained
    textstr_trained = f'''Bow-tie Score: {pattern_trained['bow_tie_score']:.3f}
Early/Middle/Late: {pattern_trained['early_importance']:.4f}/{pattern_trained['middle_importance']:.4f}/{pattern_trained['late_importance']:.4f}
Early/Middle/Late Mass: {pattern_trained['early_mass_fraction']:.3f}/{pattern_trained['middle_mass_fraction']:.3f}/{pattern_trained['late_mass_fraction']:.3f}
Sparsity: {pattern_trained['sparsity']:.3f}'''
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax2.text(0.02, 0.98, textstr_trained, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_random_vs_trained_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed comparison
    comparison_file = output_dir / f'{model_name}_random_vs_trained_analysis.txt'
    with open(comparison_file, 'w') as f:
        f.write(f"Random vs Trained Network Fisher Analysis - {model_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("RANDOM NETWORK ANALYSIS:\n")
        f.write(f"  Eigenvalue: {S_random[0]:.6f}\n")
        f.write(f"  Bow-tie Score: {pattern_random['bow_tie_score']:.6f}\n")
        f.write(f"  Early/Middle/Late Importance: {pattern_random['early_importance']:.6f}/{pattern_random['middle_importance']:.6f}/{pattern_random['late_importance']:.6f}\n")
        f.write(f"  Early/Middle/Late Mass: {pattern_random['early_mass_fraction']:.4f}/{pattern_random['middle_mass_fraction']:.4f}/{pattern_random['late_mass_fraction']:.4f}\n")
        f.write(f"  Sparsity: {pattern_random['sparsity']:.4f}\n\n")
        
        f.write("TRAINED NETWORK ANALYSIS:\n")
        f.write(f"  Eigenvalue: {S_trained[0]:.6f}\n")
        f.write(f"  Bow-tie Score: {pattern_trained['bow_tie_score']:.6f}\n")
        f.write(f"  Early/Middle/Late Importance: {pattern_trained['early_importance']:.6f}/{pattern_trained['middle_importance']:.6f}/{pattern_trained['late_importance']:.6f}\n")
        f.write(f"  Early/Middle/Late Mass: {pattern_trained['early_mass_fraction']:.4f}/{pattern_trained['middle_mass_fraction']:.4f}/{pattern_trained['late_mass_fraction']:.4f}\n")
        f.write(f"  Sparsity: {pattern_trained['sparsity']:.4f}\n\n")
        
        f.write("COMPARISON:\n")
        f.write(f"  Bow-tie Score Difference: {abs(pattern_trained['bow_tie_score'] - pattern_random['bow_tie_score']):.6f}\n")
        f.write(f"  Early Importance Difference: {abs(pattern_trained['early_importance'] - pattern_random['early_importance']):.6f}\n")
        f.write(f"  Late Importance Difference: {abs(pattern_trained['late_importance'] - pattern_random['late_importance']):.6f}\n")
        f.write(f"  Eigenvalue Ratio (Trained/Random): {S_trained[0] / S_random[0]:.6f}\n")
        
        # Interpret results
        bow_tie_diff = abs(pattern_trained['bow_tie_score'] - pattern_random['bow_tie_score'])
        if bow_tie_diff < 1.0:
            f.write(f"\nINTERPRETATION: Pattern likely ARCHITECTURAL (small difference: {bow_tie_diff:.3f})\n")
        elif pattern_trained['bow_tie_score'] > pattern_random['bow_tie_score']:
            f.write(f"\nINTERPRETATION: Pattern ENHANCED by training (difference: {bow_tie_diff:.3f})\n")
        else:
            f.write(f"\nINTERPRETATION: Pattern REDUCED by training (difference: {bow_tie_diff:.3f})\n")
    
    # Print summary
    print(f"\nRANDOM VS TRAINED COMPARISON:")
    print(f"  Random bow-tie score: {pattern_random['bow_tie_score']:.3f}")
    print(f"  Trained bow-tie score: {pattern_trained['bow_tie_score']:.3f}")
    print(f"  Difference: {abs(pattern_trained['bow_tie_score'] - pattern_random['bow_tie_score']):.3f}")
    print(f"  Random eigenvalue: {S_random[0]:.6f}")
    print(f"  Trained eigenvalue: {S_trained[0]:.6f}")
    
    # Interpretation
    bow_tie_diff = abs(pattern_trained['bow_tie_score'] - pattern_random['bow_tie_score'])
    if bow_tie_diff < 1.0:
        print(f"\n🏗️  CONCLUSION: Pattern likely ARCHITECTURAL (exists in random network)")
    elif pattern_trained['bow_tie_score'] > pattern_random['bow_tie_score']:
        print(f"\n🎯  CONCLUSION: Pattern ENHANCED by training")
    else:
        print(f"\n📉  CONCLUSION: Pattern REDUCED by training")
    
    return pattern_random, pattern_trained


def load_model_by_name(model_name):
    """
    Load a model by name without interactive selection.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (model, model_name, data_loader)
    """
    from utils.model_loader import load_model_from_trainer
    
    # Map model names to their trainer module paths
    trainer_path = f'trainers.specific_trainers.{model_name.lower()}'
    
    try:
        model, loaded_name, data_loader = load_model_from_trainer(trainer_path)
        return model, loaded_name, data_loader
    except Exception as e:
        raise Exception(f"Failed to load model '{model_name}': {e}")


def get_available_models():
    """
    Get list of available trained models.
    
    Returns:
        List of available model names
    """
    outputs_dir = Path('trainers') / 'outputs'
    
    if not outputs_dir.exists():
        return []
    
    available_models = []
    for model_dir in outputs_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        checkpoint_dir = model_dir / 'checkpoints'
        if not checkpoint_dir.exists():
            continue
            
        # Check if there are any checkpoint files
        checkpoint_files = list(checkpoint_dir.glob('*.pt'))
        if checkpoint_files:
            available_models.append(model_dir.name)
    
    return sorted(available_models)


def main():
    """
    Main function to analyze random vs trained network Fisher patterns.
    """
    parser = argparse.ArgumentParser(description='Analyze random vs trained network Fisher Information patterns')
    parser.add_argument('--model', type=str, help='Model name to analyze (optional, will list available if not provided)')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        available_models = get_available_models()
        print("Available models:")
        for i, model_name in enumerate(available_models):
            print(f"  {i}: {model_name}")
        return
    
    # If no model specified, list available and exit
    if not args.model:
        available_models = get_available_models()
        print("Available models:")
        for i, model_name in enumerate(available_models):
            print(f"  {i}: {model_name}")
        print(f"\nUsage: python {sys.argv[0]} --model MODEL_NAME")
        print(f"   or: python {sys.argv[0]} --list-models")
        return
    
    try:
        # Load specified model
        print(f"Loading model: {args.model}")
        model, model_name, data_loader = load_model_by_name(args.model)
        
        # Get train data loader
        train_loader = data_loader.get_train_loader()
        
        print(f"\nAnalyzing random vs trained comparison for: {model_name}")
        
        # Create output directory
        output_dir = Path(f'model_interpretation/outputs/fisher_analysis/{model_name}/random_vs_trained')
        
        # Perform comparison
        pattern_random, pattern_trained = compare_trained_vs_random(
            model, train_loader, model_name, output_dir
        )
        
        print(f"\nResults saved to: {output_dir}")
        
        # Final research implications
        bow_tie_diff = abs(pattern_trained['bow_tie_score'] - pattern_random['bow_tie_score'])
        print(f"\n🔬 RESEARCH IMPLICATIONS:")
        if bow_tie_diff < 1.0:
            print(f"   • Bow-tie pattern is ARCHITECTURAL, not learned")
            print(f"   • Suggests fundamental property of network structure")
            print(f"   • Could apply to many networks with similar architecture")
        else:
            print(f"   • Bow-tie pattern is LEARNED during training")
            print(f"   • Training process shapes parameter sensitivity")
            print(f"   • Pattern emergence is part of learning dynamics")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        available_models = get_available_models()
        print(f"\nAvailable models: {', '.join(available_models[:5])}{'...' if len(available_models) > 5 else ''}")


if __name__ == "__main__":
    main()
