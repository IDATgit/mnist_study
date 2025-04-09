import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import MNISTDataLoader
from models.specific_models.ShiftInvariantCNN import ShiftInvariantCNN
from models.specific_models.LinearModel import LinearModel

def calculate_fisher_information(model, data_loader, num_samples=1000):
    """
    Calculate the empirical Fisher Information Matrix using the outer product form.
    FIM = E[∇log p(y|x,θ)∇log p(y|x,θ)^T]
    """
    model.train()  # Set to training mode for gradient computation
    num_params = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device
    
    # Initialize FIM as zero matrix
    fisher_info = torch.zeros((num_params, num_params), device=device)
    
    # Calculate FIM using gradients
    for batch_idx, (data, _) in enumerate(data_loader):
            
        data = data.to(device)
        outputs = model(data)
        probs = torch.softmax(outputs, dim=1)
        log_probs = torch.log(probs)
        
        # Compute gradients for each sample in the batch
        for i in range(data.size(0)):
            # Compute gradients for each class
            for j in range(probs.size(1)):
                # Get score for this sample and class
                score = log_probs[i, j].clone()
                prob = probs[i, j].item()  # Get scalar value
                
                
                # Compute gradient with respect to parameters
                score.backward(retain_graph=True)
                
                # Get flattened gradient and detach
                grad = torch.cat([p.grad.detach().view(-1) for p in model.parameters()])
                
                # Add outer product to Fisher Information Matrix
                fisher_info.add_(torch.outer(grad, grad) * prob)
                
                # Zero gradients
                model.zero_grad()
                
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Analyzed {batch_idx + 1} batches...")
    
    # Average over samples
    fisher_info /= (num_samples * data.size(0))
    
    return fisher_info.cpu().numpy()

def analyze_fisher_information(fisher_info, model_name, output_dir):
    """
    Analyze the Fisher Information Matrix through spectral decomposition.
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate eigenvalues
    eigenvalues, eigenvectors = eigh(fisher_info)
    
    # Sort eigenvalues in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Plot eigenvalue distribution
    plt.figure(figsize=(10, 6))
    plt.hist(eigenvalues, bins=50, density=True)
    plt.title(f'Eigenvalue Distribution of Fisher Information Matrix\n{model_name}')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(output_dir / f'{model_name}_fisher_eigenvalues.png')
    plt.close()
    
    # Calculate statistics
    stats = {
        'max_eigenvalue': eigenvalues[0],
        'min_eigenvalue': eigenvalues[-1],
        'mean_eigenvalue': np.mean(eigenvalues),
        'median_eigenvalue': np.median(eigenvalues),
        'std_eigenvalue': np.std(eigenvalues),
        'condition_number': eigenvalues[0] / eigenvalues[-1],
        'effective_rank': np.sum(eigenvalues) / eigenvalues[0]
    }
    
    # Save statistics
    with open(output_dir / f'{model_name}_fisher_stats.txt', 'w') as f:
        for key, value in stats.items():
            f.write(f'{key}: {value:.6f}\n')
    
    return stats

def main():
    # Initialize data loader
    data_loader = MNISTDataLoader(batch_size=64, preload_gpu=True)
    train_loader = data_loader.get_train_loader()
    
    # Model to analyze
    model = LinearModel()
    model_name = 'LinearModel'
    
    # Output directory
    output_dir = Path('model_interpretation/outputs/fisher_analysis')
    
    print(f"\nAnalyzing {model_name}...")
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Calculate Fisher Information Matrix
    fisher_info = calculate_fisher_information(model, train_loader)
    
    # Analyze and save results
    stats = analyze_fisher_information(fisher_info, model_name, output_dir)
    
    # Print summary statistics
    print(f"Fisher Information Analysis for {model_name}:")
    print(f"Max eigenvalue: {stats['max_eigenvalue']:.6f}")
    print(f"Min eigenvalue: {stats['min_eigenvalue']:.6f}")
    print(f"Condition number: {stats['condition_number']:.6f}")
    print(f"Effective rank: {stats['effective_rank']:.6f}")

if __name__ == "__main__":
    main() 