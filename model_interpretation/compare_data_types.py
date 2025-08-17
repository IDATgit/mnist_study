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
from visualize_eigenvectors import quantify_bow_tie_pattern, load_eigenvector_data


def compare_data_types(model_names, data_type='train', output_dir='model_interpretation/outputs/data_type_comparison'):
    """
    Compare bow-tie patterns across different data types (real vs random labels/images).
    
    Args:
        model_names: List of model names to compare
        data_type: 'train' or 'test'
        output_dir: Directory to save comparison results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print(f"Comparing bow-tie patterns across data types ({data_type} data):")
    print("=" * 60)
    
    # Analyze each model
    for model_name in model_names:
        try:
            print(f"\nAnalyzing {model_name}...")
            U, S, stats = load_eigenvector_data(model_name, data_type)
            
            # Get first eigenvector and quantify pattern
            first_eigenvector = U[:, 0]
            num_params = len(first_eigenvector)
            pattern_metrics = quantify_bow_tie_pattern(first_eigenvector, num_params)
            
            results[model_name] = {
                'bow_tie_score': pattern_metrics['bow_tie_score'],
                'early_importance': pattern_metrics['early_importance'],
                'middle_importance': pattern_metrics['middle_importance'],
                'late_importance': pattern_metrics['late_importance'],
                'early_mass_fraction': pattern_metrics['early_mass_fraction'],
                'late_mass_fraction': pattern_metrics['late_mass_fraction'],
                'sparsity': pattern_metrics['sparsity'],
                'eigenvalue': S[0],
                'num_params': num_params
            }
            
            print(f"  Bow-tie score: {pattern_metrics['bow_tie_score']:.3f}")
            print(f"  Early/Late mass: {pattern_metrics['early_mass_fraction']:.3f}/{pattern_metrics['late_mass_fraction']:.3f}")
            
        except FileNotFoundError as e:
            print(f"  ❌ No Fisher data found for {model_name}")
            results[model_name] = None
        except Exception as e:
            print(f"  ❌ Error analyzing {model_name}: {e}")
            results[model_name] = None
    
    # Create comparison visualization
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) < 2:
        print(f"\nInsufficient data for comparison (only {len(valid_results)} valid models)")
        return results
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    model_names_clean = list(valid_results.keys())
    bow_tie_scores = [valid_results[m]['bow_tie_score'] for m in model_names_clean]
    early_mass = [valid_results[m]['early_mass_fraction'] for m in model_names_clean]
    late_mass = [valid_results[m]['late_mass_fraction'] for m in model_names_clean]
    sparsity = [valid_results[m]['sparsity'] for m in model_names_clean]
    
    # Clean up model names for display
    display_names = []
    for name in model_names_clean:
        if 'random_labels' in name:
            display_names.append('Random Labels')
        elif 'random_images' in name:
            display_names.append('Random Images')
        elif 'random' not in name:
            display_names.append('Real Data')
        else:
            display_names.append(name.replace('small_convnet_', '').replace('_', ' ').title())
    
    # Plot bow-tie scores
    bars1 = ax1.bar(display_names, bow_tie_scores, color=['blue', 'red', 'green', 'orange'][:len(display_names)])
    ax1.set_title(f'Bow-Tie Pattern Strength Comparison\n({data_type.title()} Data)')
    ax1.set_ylabel('Bow-Tie Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=2.0, color='black', linestyle='--', alpha=0.5, label='Strong Pattern Threshold')
    ax1.legend()
    
    # Add value labels on bars
    for bar, score in zip(bars1, bow_tie_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom')
    
    # Plot early vs late dominance
    x_pos = np.arange(len(display_names))
    width = 0.35
    ax2.bar(x_pos - width/2, early_mass, width, label='Early Region (10%)', alpha=0.8)
    ax2.bar(x_pos + width/2, late_mass, width, label='Late Region (10%)', alpha=0.8)
    ax2.set_title('Parameter Importance Distribution')
    ax2.set_ylabel('Mass Fraction')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(display_names, rotation=45)
    ax2.legend()
    
    # Plot sparsity
    bars3 = ax3.bar(display_names, sparsity, color=['blue', 'red', 'green', 'orange'][:len(display_names)])
    ax3.set_title('Eigenvector Sparsity')
    ax3.set_ylabel('Sparsity (fraction near zero)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, sparse in zip(bars3, sparsity):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{sparse:.2f}', ha='center', va='bottom')
    
    # Summary table
    ax4.axis('tight')
    ax4.axis('off')
    table_data = []
    headers = ['Model', 'Bow-Tie Score', 'Early Mass', 'Late Mass', 'Sparsity']
    
    for i, name in enumerate(model_names_clean):
        table_data.append([
            display_names[i],
            f"{valid_results[name]['bow_tie_score']:.3f}",
            f"{valid_results[name]['early_mass_fraction']:.3f}",
            f"{valid_results[name]['late_mass_fraction']:.3f}",
            f"{valid_results[name]['sparsity']:.3f}"
        ])
    
    table = ax4.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Summary Statistics', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'data_type_comparison_{data_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    results_file = output_dir / f'data_type_comparison_{data_type}.txt'
    with open(results_file, 'w') as f:
        f.write(f"Data Type Comparison Analysis ({data_type} data)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("RESEARCH QUESTION: Does bow-tie pattern depend on data meaningfulness?\n\n")
        
        for model_name in model_names_clean:
            result = valid_results[model_name]
            f.write(f"{model_name}:\n")
            f.write(f"  Bow-tie Score: {result['bow_tie_score']:.6f}\n")
            f.write(f"  Early/Middle/Late Importance: {result['early_importance']:.6f}/{result['middle_importance']:.6f}/{result['late_importance']:.6f}\n")
            f.write(f"  Early/Late Mass Fraction: {result['early_mass_fraction']:.4f}/{result['late_mass_fraction']:.4f}\n")
            f.write(f"  Sparsity: {result['sparsity']:.4f}\n")
            f.write(f"  First Eigenvalue: {result['eigenvalue']:.6f}\n")
            f.write(f"  Parameters: {result['num_params']}\n\n")
        
        # Analysis
        real_data_score = None
        random_labels_score = None
        random_images_score = None
        
        for name in model_names_clean:
            score = valid_results[name]['bow_tie_score']
            if 'random_labels' in name:
                random_labels_score = score
            elif 'random_images' in name:
                random_images_score = score
            elif 'random' not in name:
                real_data_score = score
        
        f.write("INTERPRETATION:\n")
        if real_data_score and random_labels_score:
            if abs(real_data_score - random_labels_score) < 1.0:
                f.write("• Random labels show similar pattern → Bow-tie may not require meaningful supervision\n")
            else:
                f.write("• Random labels show different pattern → Meaningful supervision affects bow-tie formation\n")
        
        if real_data_score and random_images_score:
            if abs(real_data_score - random_images_score) < 1.0:
                f.write("• Random images show similar pattern → Bow-tie may not require meaningful input structure\n")
            else:
                f.write("• Random images show different pattern → Input structure affects bow-tie formation\n")
    
    print(f"\nComparison results saved to: {output_dir}")
    return results


def main():
    """
    Main function to compare bow-tie patterns across different data types.
    """
    parser = argparse.ArgumentParser(description='Compare Fisher Information patterns across data types')
    parser.add_argument('--models', nargs='+', help='Model names to compare')
    parser.add_argument('--data-type', type=str, choices=['train', 'test'], default='train', help='Data type to analyze')
    parser.add_argument('--auto', action='store_true', help='Automatically find and compare similar models')
    
    args = parser.parse_args()
    
    if args.auto:
        # Auto-detect models for comparison
        from visualize_eigenvectors import get_available_fisher_models
        available_models = get_available_fisher_models()
        
        # Find models that might be related
        base_models = []
        for model in available_models:
            if 'small_convnet' in model and '10k' in model:
                base_models.append(model)
        
        if not base_models:
            print("No suitable models found for automatic comparison")
            return
        
        print(f"Auto-detected models for comparison: {base_models}")
        model_names = base_models
    elif args.models:
        model_names = args.models
    else:
        print("Please specify models to compare with --models or use --auto")
        print("Example: python compare_data_types.py --models small_convnet_10k small_convnet_random_labels_10k")
        return
    
    # Run comparison
    results = compare_data_types(model_names, args.data_type)
    
    # Print summary
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        print(f"\n🔬 RESEARCH SUMMARY ({args.data_type} data):")
        for name, result in valid_results.items():
            pattern_strength = "STRONG" if result['bow_tie_score'] > 2.0 else "WEAK"
            print(f"  {name}: {result['bow_tie_score']:.3f} ({pattern_strength})")
    

if __name__ == "__main__":
    main()
