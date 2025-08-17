import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import argparse

# Add the project root to the path
sys.path.append('.')
sys.path.append('..')

def get_layer_parameter_mapping(model):
    """
    Create mapping of parameter indices to layers for DimensionReductionFC.
    
    Returns:
        dict: Mapping of layer names to parameter index ranges
    """
    layer_map = {}
    param_idx = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        layer_map[name] = {
            'start': param_idx,
            'end': param_idx + param_count,
            'count': param_count,
            'shape': list(param.shape),
            'compression_ratio': None  # Will calculate later
        }
        param_idx += param_count
    
    # Calculate compression ratios for FC layers
    for name, info in layer_map.items():
        if 'fc_layers' in name and 'weight' in name:
            shape = info['shape']
            if len(shape) == 2:  # [out_features, in_features]
                out_features, in_features = shape
                compression_ratio = in_features / out_features if out_features > 0 else 1.0
                info['compression_ratio'] = compression_ratio
                info['reduction_percentage'] = (1 - 1/compression_ratio) * 100 if compression_ratio > 1 else 0
    
    return layer_map

def categorize_layers_fc(layer_map):
    """
    Categorize layers for the fully connected DimensionReductionFC model.
    """
    categories = {
        'fc_weights': [],
        'fc_biases': [],
        'bn_weights': [],
        'bn_biases': [],
        'all_weights': [],
        'all_biases': []
    }
    
    for name, info in layer_map.items():
        if 'fc_layers' in name:
            if 'weight' in name:
                categories['fc_weights'].append((name, info))
                categories['all_weights'].append((name, info))
            elif 'bias' in name:
                categories['fc_biases'].append((name, info))
                categories['all_biases'].append((name, info))
        elif 'batch_norms' in name:
            if 'weight' in name:
                categories['bn_weights'].append((name, info))
                categories['all_weights'].append((name, info))
            elif 'bias' in name:
                categories['bn_biases'].append((name, info))
                categories['all_biases'].append((name, info))
    
    return categories

def analyze_layer_contributions_fc(model_name, output_base_dir='model_interpretation/outputs/fisher_analysis'):
    """
    Analyze layer-wise contributions to Fisher Information for DimensionReductionFC model.
    """
    
    # Load the fisher analysis results
    fisher_dir = Path(output_base_dir) / model_name
    
    # Load U, S matrices (try RSVD files first, then fallback to regular)
    U_file = fisher_dir / f'train_{model_name}_U_rsvd.npy'
    S_file = fisher_dir / f'train_{model_name}_S_rsvd.npy'
    
    if not U_file.exists() or not S_file.exists():
        # Fallback to non-RSVD files
        U_file = fisher_dir / f'train_{model_name}_U.npy'
        S_file = fisher_dir / f'train_{model_name}_S.npy'
    
    if not U_file.exists() or not S_file.exists():
        print(f"Fisher analysis files not found for {model_name}")
        print(f"Expected files:")
        print(f"  {U_file}")
        print(f"  {S_file}")
        return
    
    print(f"Loading Fisher analysis results for {model_name}...")
    U = np.load(U_file)
    S = np.load(S_file)
    
    print(f"Loaded U: {U.shape}, S: {S.shape}")
    
    # Load the model to get parameter structure
    from utils.model_loader import load_model_from_trainer
    
    try:
        trainer_module_path = f"trainers.specific_trainers.{model_name}"
        model, _, _ = load_model_from_trainer(trainer_module_path)
        print(f"Loaded model: {model._get_name()}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get layer parameter mapping
    layer_map = get_layer_parameter_mapping(model)
    categories = categorize_layers_fc(layer_map)
    
    print("\nModel Architecture Analysis:")
    print("=" * 60)
    
    total_params = sum(info['count'] for info in layer_map.values())
    print(f"Total parameters: {total_params:,}")
    
    # Print layer details with compression ratios
    for name, info in layer_map.items():
        compression_str = ""
        if info['compression_ratio'] is not None:
            compression_str = f" (compression: {info['compression_ratio']:.2f}x, reduction: {info['reduction_percentage']:.1f}%)"
        print(f"{name}: {info['count']:,} params {info['shape']}{compression_str}")
    
    # Analyze first eigenvector (most significant)
    first_eigenvector = U[:, 0]
    print(f"\nAnalyzing first eigenvector (eigenvalue: {S[0]:.6f})...")
    
    # Calculate layer-wise contributions
    layer_contributions = {}
    layer_masses = {}
    
    for name, info in layer_map.items():
        start_idx = info['start']
        end_idx = info['end']
        layer_vector = first_eigenvector[start_idx:end_idx]
        
        # Calculate contribution metrics
        mass = np.sum(np.abs(layer_vector))
        mean_abs = np.mean(np.abs(layer_vector))
        max_abs = np.max(np.abs(layer_vector))
        std = np.std(layer_vector)
        
        layer_contributions[name] = {
            'mass': mass,
            'mean_abs': mean_abs,
            'max_abs': max_abs,
            'std': std,
            'mass_fraction': mass / np.sum(np.abs(first_eigenvector)),
            'param_count': info['count'],
            'compression_ratio': info.get('compression_ratio', None)
        }
        layer_masses[name] = mass
    
    # Sort by mass fraction
    sorted_layers = sorted(layer_contributions.items(), key=lambda x: x[1]['mass_fraction'], reverse=True)
    
    print("\nLayer-wise Fisher Information Mass (First Eigenvector):")
    print("=" * 80)
    print(f"{'Layer':<35} {'Mass Frac':<10} {'Compression':<12} {'Params':<10} {'Mean|v|':<10}")
    print("-" * 80)
    
    for name, contrib in sorted_layers:
        compression_str = f"{contrib['compression_ratio']:.2f}x" if contrib['compression_ratio'] else "N/A"
        print(f"{name:<35} {contrib['mass_fraction']:.4f}     {compression_str:<12} {contrib['param_count']:<10,} {contrib['mean_abs']:.6f}")
    
    # Analyze bottleneck hypothesis
    print("\n🔬 BOTTLENECK ANALYSIS:")
    print("=" * 50)
    
    # Find layers with highest compression ratios
    fc_layers = [(name, info) for name, info in layer_contributions.items() 
                 if 'fc_layers' in name and 'weight' in name and info['compression_ratio']]
    
    if fc_layers:
        # Sort by compression ratio
        fc_layers_by_compression = sorted(fc_layers, key=lambda x: x[1]['compression_ratio'], reverse=True)
        
        print("FC Layers by Compression Ratio:")
        for name, info in fc_layers_by_compression:
            print(f"  {name}: {info['compression_ratio']:.2f}x compression → {info['mass_fraction']:.4f} Fisher mass")
        
        # Test bottleneck hypothesis: higher compression = higher Fisher sensitivity?
        compressions = [info['compression_ratio'] for _, info in fc_layers_by_compression]
        fisher_masses = [info['mass_fraction'] for _, info in fc_layers_by_compression]
        
        # Simple correlation
        correlation = np.corrcoef(compressions, fisher_masses)[0, 1]
        print(f"\nCorrelation between compression ratio and Fisher mass: {correlation:.4f}")
        
        if correlation > 0.5:
            print("✅ STRONG POSITIVE correlation: Higher compression → Higher Fisher sensitivity")
        elif correlation > 0.2:
            print("✅ MODERATE POSITIVE correlation: Some support for bottleneck hypothesis")
        elif correlation < -0.2:
            print("❌ NEGATIVE correlation: Evidence against bottleneck hypothesis")
        else:
            print("❓ WEAK correlation: Inconclusive evidence")
    
    # Create visualization
    create_layer_analysis_plots(model_name, layer_contributions, fisher_dir)
    
    # Save detailed analysis
    save_layer_analysis(model_name, layer_contributions, layer_map, fisher_dir)
    
    return layer_contributions

def create_layer_analysis_plots(model_name, layer_contributions, output_dir):
    """
    Create visualization plots for layer analysis.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get FC weight layers only
    fc_layers = {name: info for name, info in layer_contributions.items() 
                 if 'fc_layers' in name and 'weight' in name}
    
    if fc_layers:
        names = list(fc_layers.keys())
        # Simplify names for plotting
        simple_names = [name.replace('fc_layers.', 'FC').replace('.weight', '') for name in names]
        masses = [fc_layers[name]['mass_fraction'] for name in names]
        compressions = [fc_layers[name]['compression_ratio'] for name in names]
        
        # Plot 1: Fisher mass by layer
        ax1.bar(simple_names, masses, alpha=0.7, color='steelblue')
        ax1.set_title('Fisher Information Mass by Layer')
        ax1.set_ylabel('Mass Fraction')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Compression ratio by layer
        ax2.bar(simple_names, compressions, alpha=0.7, color='orange')
        ax2.set_title('Compression Ratio by Layer')
        ax2.set_ylabel('Compression Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Compression vs Fisher mass scatter
        ax3.scatter(compressions, masses, s=100, alpha=0.7, color='red')
        for i, name in enumerate(simple_names):
            ax3.annotate(name, (compressions[i], masses[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax3.set_xlabel('Compression Ratio')
        ax3.set_ylabel('Fisher Mass Fraction')
        ax3.set_title('Compression vs Fisher Sensitivity')
        
        # Add correlation line
        if len(compressions) > 1:
            z = np.polyfit(compressions, masses, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(compressions), max(compressions), 100)
            ax3.plot(x_line, p(x_line), "r--", alpha=0.8)
            
            correlation = np.corrcoef(compressions, masses)[0, 1]
            ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax3.transAxes, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 4: All parameters mass distribution
    all_names = list(layer_contributions.keys())
    all_masses = [layer_contributions[name]['mass_fraction'] for name in all_names]
    
    # Simplify names and sort by mass
    sorted_items = sorted(zip(all_names, all_masses), key=lambda x: x[1], reverse=True)
    sorted_names = [name for name, _ in sorted_items]
    sorted_masses = [mass for _, mass in sorted_items]
    
    # Take top 10 for readability
    top_names = sorted_names[:10]
    top_masses = sorted_masses[:10]
    simple_top_names = [name.replace('fc_layers.', 'FC').replace('batch_norms.', 'BN').replace('.weight', '.w').replace('.bias', '.b') 
                       for name in top_names]
    
    ax4.barh(simple_top_names, top_masses, alpha=0.7, color='green')
    ax4.set_title('Top 10 Parameters by Fisher Mass')
    ax4.set_xlabel('Mass Fraction')
    ax4.invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'{model_name}_layer_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Layer analysis plot saved to: {plot_path}")
    plt.close()

def save_layer_analysis(model_name, layer_contributions, layer_map, output_dir):
    """
    Save detailed layer analysis to text file.
    """
    analysis_file = output_dir / f'{model_name}_layer_analysis.txt'
    
    with open(analysis_file, 'w') as f:
        f.write(f"Layer-wise Fisher Information Analysis for {model_name}\n")
        f.write("=" * 60 + "\n\n")
        
        total_params = sum(info['count'] for info in layer_map.values())
        f.write(f"Total parameters: {total_params:,}\n\n")
        
        f.write("Model Architecture:\n")
        f.write("-" * 40 + "\n")
        for name, info in layer_map.items():
            compression_str = ""
            if info.get('compression_ratio'):
                compression_str = f" (compression: {info['compression_ratio']:.2f}x)"
            f.write(f"{name}: {info['count']:,} params {info['shape']}{compression_str}\n")
        
        f.write("\nLayer Contributions (sorted by Fisher mass):\n")
        f.write("-" * 60 + "\n")
        
        sorted_layers = sorted(layer_contributions.items(), 
                              key=lambda x: x[1]['mass_fraction'], reverse=True)
        
        for name, contrib in sorted_layers:
            f.write(f"{name}:\n")
            f.write(f"  Mass fraction: {contrib['mass_fraction']:.6f}\n")
            f.write(f"  Parameter count: {contrib['param_count']:,}\n")
            f.write(f"  Mean |v|: {contrib['mean_abs']:.6f}\n")
            if contrib['compression_ratio']:
                f.write(f"  Compression ratio: {contrib['compression_ratio']:.2f}x\n")
            f.write("\n")
    
    print(f"Detailed analysis saved to: {analysis_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze layer-wise Fisher Information contributions for DimensionReductionFC')
    parser.add_argument('--model', type=str, default='dimension_reduction_fc_10k',
                        help='Model name to analyze (default: dimension_reduction_fc_10k)')
    
    args = parser.parse_args()
    
    print(f"🔬 Analyzing layer-wise Fisher contributions for: {args.model}")
    print("Testing the dimensionality reduction bottleneck hypothesis...")
    print("=" * 70)
    
    layer_contributions = analyze_layer_contributions_fc(args.model)
    
    if layer_contributions:
        print(f"\n✅ Analysis completed for {args.model}")
        print("Results saved to model_interpretation/outputs/fisher_analysis/")
    else:
        print(f"\n❌ Analysis failed for {args.model}")

if __name__ == "__main__":
    main()
