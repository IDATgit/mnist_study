import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.append('.')
sys.path.append('..')

def get_layer_parameter_mapping(model):
    """
    Create mapping of parameter indices to layers for SmallConvNet.
    
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
            'shape': list(param.shape)
        }
        param_idx += param_count
    
    return layer_map

def categorize_layers(layer_map):
    """
    Categorize layers into meaningful groups.
    """
    categories = {
        'conv1': [],
        'conv2': [], 
        'fc1': [],
        'fc2': [],
        'weights': [],
        'biases': []
    }
    
    for name, info in layer_map.items():
        # Layer type categorization
        if 'conv' in name and '0' in name:
            categories['conv1'].append((name, info))
        elif 'conv' in name and '1' in name:
            categories['conv2'].append((name, info))
        elif 'fc' in name and '0' in name:
            categories['fc1'].append((name, info))
        elif 'fc' in name and '1' in name:
            categories['fc2'].append((name, info))
        
        # Parameter type categorization
        if 'weight' in name:
            categories['weights'].append((name, info))
        elif 'bias' in name:
            categories['biases'].append((name, info))
    
    return categories

def analyze_layer_contributions(eigenvector, layer_map):
    """
    Analyze how much each layer contributes to the eigenvector.
    """
    layer_contributions = {}
    
    for name, info in layer_map.items():
        start_idx = info['start']
        end_idx = info['end']
        layer_params = eigenvector[start_idx:end_idx]
        
        layer_contributions[name] = {
            'mean_abs': np.mean(np.abs(layer_params)),
            'max_abs': np.max(np.abs(layer_params)),
            'std': np.std(layer_params),
            'mass_fraction': np.sum(np.abs(layer_params)) / np.sum(np.abs(eigenvector)),
            'param_count': len(layer_params),
            'indices': (start_idx, end_idx)
        }
    
    return layer_contributions

def create_layer_visualization(eigenvector, layer_map, model_name, data_type, output_dir):
    """
    Create detailed visualization of layer-wise contributions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get layer contributions
    contributions = analyze_layer_contributions(eigenvector, layer_map)
    categories = categorize_layers(layer_map)
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Parameter-wise eigenvector plot with layer boundaries
    param_indices = np.arange(len(eigenvector))
    ax1.plot(param_indices, eigenvector, 'b-', linewidth=0.8, alpha=0.8)
    
    # Add vertical lines for layer boundaries
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    color_idx = 0
    for name, info in layer_map.items():
        if 'weight' in name:  # Only show weight boundaries for clarity
            ax1.axvline(info['start'], color=colors[color_idx % len(colors)], 
                       linestyle='--', alpha=0.7, label=name)
            color_idx += 1
    
    ax1.set_title(f'Eigenvector with Layer Boundaries\n{model_name} - {data_type.title()} Data')
    ax1.set_xlabel('Parameter Index')
    ax1.set_ylabel('Eigenvector Value')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Layer mass contributions
    layer_names = list(contributions.keys())
    layer_masses = [contributions[name]['mass_fraction'] for name in layer_names]
    
    bars = ax2.bar(range(len(layer_names)), layer_masses, alpha=0.8)
    ax2.set_title('Mass Fraction by Layer')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Mass Fraction')
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels([name.replace('features.', '').replace('classifier.', '') 
                        for name in layer_names], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, mass in zip(bars, layer_masses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mass:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Weights vs Biases comparison
    weight_mass = sum([contributions[name]['mass_fraction'] 
                      for name, _ in categories['weights']])
    bias_mass = sum([contributions[name]['mass_fraction'] 
                    for name, _ in categories['biases']])
    
    ax3.bar(['Weights', 'Biases'], [weight_mass, bias_mass], 
           color=['blue', 'red'], alpha=0.8)
    ax3.set_title('Weights vs Biases Contribution')
    ax3.set_ylabel('Total Mass Fraction')
    
    for i, (label, mass) in enumerate([('Weights', weight_mass), ('Biases', bias_mass)]):
        ax3.text(i, mass, f'{mass:.3f}', ha='center', va='bottom')
    
    # 4. Layer type comparison
    layer_type_masses = {}
    for layer_type in ['conv1', 'conv2', 'fc1', 'fc2']:
        if categories[layer_type]:
            layer_type_masses[layer_type] = sum([
                contributions[name]['mass_fraction'] 
                for name, _ in categories[layer_type]
            ])
    
    if layer_type_masses:
        ax4.bar(layer_type_masses.keys(), layer_type_masses.values(), alpha=0.8)
        ax4.set_title('Contribution by Layer Type')
        ax4.set_ylabel('Total Mass Fraction')
        
        for layer_type, mass in layer_type_masses.items():
            ax4.text(list(layer_type_masses.keys()).index(layer_type), mass, 
                    f'{mass:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{data_type}_{model_name}_layer_decomposition.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    return contributions, categories

def save_detailed_analysis(contributions, categories, layer_map, model_name, data_type, output_dir):
    """
    Save detailed text analysis of layer contributions.
    """
    output_dir = Path(output_dir)
    
    with open(output_dir / f'{data_type}_{model_name}_layer_analysis.txt', 'w') as f:
        f.write(f"Layer-wise Fisher Eigenvector Analysis\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Data: {data_type}\n")
        f.write("=" * 60 + "\n\n")
        
        # Individual layer analysis
        f.write("INDIVIDUAL LAYER CONTRIBUTIONS:\n")
        sorted_layers = sorted(contributions.items(), 
                             key=lambda x: x[1]['mass_fraction'], reverse=True)
        
        for name, contrib in sorted_layers:
            f.write(f"\n{name}:\n")
            f.write(f"  Mass fraction: {contrib['mass_fraction']:.6f}\n")
            f.write(f"  Mean |value|: {contrib['mean_abs']:.6f}\n")
            f.write(f"  Max |value|: {contrib['max_abs']:.6f}\n")
            f.write(f"  Std deviation: {contrib['std']:.6f}\n")
            f.write(f"  Parameter count: {contrib['param_count']}\n")
            f.write(f"  Index range: {contrib['indices'][0]}-{contrib['indices'][1]}\n")
        
        # Category analysis
        f.write(f"\n\nCATEGORY ANALYSIS:\n")
        
        # Weights vs Biases
        weight_mass = sum([contributions[name]['mass_fraction'] 
                          for name, _ in categories['weights']])
        bias_mass = sum([contributions[name]['mass_fraction'] 
                        for name, _ in categories['biases']])
        
        f.write(f"\nWeights vs Biases:\n")
        f.write(f"  Total weight contribution: {weight_mass:.6f}\n")
        f.write(f"  Total bias contribution: {bias_mass:.6f}\n")
        f.write(f"  Weight/Bias ratio: {weight_mass/bias_mass:.3f}\n")
        
        # Layer types
        f.write(f"\nLayer Types:\n")
        for layer_type in ['conv1', 'conv2', 'fc1', 'fc2']:
            if categories[layer_type]:
                total_mass = sum([contributions[name]['mass_fraction'] 
                                for name, _ in categories[layer_type]])
                param_count = sum([info['count'] for _, info in categories[layer_type]])
                f.write(f"  {layer_type}: mass={total_mass:.6f}, params={param_count}\n")
        
        # Insights
        f.write(f"\n\nKEY INSIGHTS:\n")
        
        # Most important layer
        top_layer = sorted_layers[0]
        f.write(f"• Most important layer: {top_layer[0]} ({top_layer[1]['mass_fraction']:.3f} of total mass)\n")
        
        # Early vs Late analysis
        early_layers = [name for name in contributions.keys() if ('conv' in name or 'features.0' in name)]
        late_layers = [name for name in contributions.keys() if ('fc' in name and ('1' in name or 'classifier' in name))]
        
        early_mass = sum([contributions[name]['mass_fraction'] for name in early_layers])
        late_mass = sum([contributions[name]['mass_fraction'] for name in late_layers])
        
        f.write(f"• Early layers total mass: {early_mass:.6f}\n")
        f.write(f"• Late layers total mass: {late_mass:.6f}\n")
        f.write(f"• Early/Late ratio: {early_mass/late_mass:.3f}\n")
        
        if weight_mass > 0.8:
            f.write(f"• Weights dominate over biases ({weight_mass:.1%} vs {bias_mass:.1%})\n")
        
        if early_mass > 0.4:
            f.write(f"• Strong early layer contribution confirms bow-tie pattern\n")
        if late_mass > 0.2:
            f.write(f"• Significant late layer contribution confirms bow-tie pattern\n")

def main():
    """
    Perform layer-wise analysis on small_convnet_10k.
    """
    model_name = 'small_convnet_10k'
    
    # Load the model to get parameter structure
    sys.path.append('models/specific_models')
    from SmallConvNet import SmallConvNet
    
    model = SmallConvNet()
    layer_map = get_layer_parameter_mapping(model)
    
    print(f"🔬 LAYER-WISE FISHER ANALYSIS: {model_name}")
    print("=" * 60)
    
    print("\nModel Architecture:")
    total_params = 0
    for name, info in layer_map.items():
        print(f"  {name}: {info['shape']} -> {info['count']} parameters")
        total_params += info['count']
    print(f"  Total: {total_params} parameters")
    
    # Try to load Fisher data for both train and test
    data_types = ['train', 'test']
    base_path = f'model_interpretation/outputs/fisher_analysis/{model_name}'
    
    for data_type in data_types:
        try:
            # Load eigenvector data
            U = np.load(f'{base_path}/{data_type}_{model_name}_U_rsvd.npy')
            S = np.load(f'{base_path}/{data_type}_{model_name}_S_rsvd.npy')
            
            print(f"\n📊 ANALYZING {data_type.upper()} DATA:")
            print(f"First eigenvalue: {S[0]:.6f}")
            
            # Get first eigenvector
            first_eigenvector = U[:, 0]
            
            # Create output directory
            output_dir = Path(f'model_interpretation/outputs/fisher_analysis/{model_name}/layer_analysis')
            
            # Perform analysis
            contributions, categories = create_layer_visualization(
                first_eigenvector, layer_map, model_name, data_type, output_dir
            )
            
            # Save detailed analysis
            save_detailed_analysis(
                contributions, categories, layer_map, model_name, data_type, output_dir
            )
            
            # Print key findings
            sorted_layers = sorted(contributions.items(), 
                                 key=lambda x: x[1]['mass_fraction'], reverse=True)
            
            print(f"Top 3 contributing layers:")
            for i, (name, contrib) in enumerate(sorted_layers[:3]):
                print(f"  {i+1}. {name}: {contrib['mass_fraction']:.3f} ({contrib['param_count']} params)")
            
            # Summary insights
            weight_mass = sum([contributions[name]['mass_fraction'] 
                              for name, _ in categories['weights']])
            bias_mass = sum([contributions[name]['mass_fraction'] 
                            for name, _ in categories['biases']])
            
            print(f"Weights contribute {weight_mass:.3f}, Biases contribute {bias_mass:.3f}")
            
        except FileNotFoundError:
            print(f"❌ No Fisher data found for {data_type} data")
        except Exception as e:
            print(f"❌ Error analyzing {data_type} data: {e}")
    
    print(f"\n✅ Analysis complete! Results saved to:")
    print(f"   model_interpretation/outputs/fisher_analysis/{model_name}/layer_analysis/")

if __name__ == "__main__":
    main()
