import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import cupy as cp
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import MNISTDataLoader
from models.specific_models.ShiftInvariantCNN import ShiftInvariantCNN
from models.specific_models.StandardFullyConnected import StandardFullyConnected
from models.specific_models.LinearModel import LinearModel


def calculate_fisher_information(model, data_loader):
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
    nof_samples = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        outputs = model(data)
        probs = torch.softmax(outputs, dim=1)
        log_probs = torch.log(probs)
        nof_samples += data.size(0)
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
            print(f"Analyzed {(batch_idx + 1) * data.size(0)} samples...")
    
    # Average over samples
    print(f"Analyzed {nof_samples} samples. (full training set)")
    fisher_info /= (nof_samples * data.size(0))
    
    return fisher_info.cpu().numpy()

def analyze_fisher_information(fisher_info, model, model_name, output_dir):
    """
    Analyze the Fisher Information Matrix through spectral decomposition.
    
    Args:
        fisher_info: The Fisher Information Matrix
        model: The PyTorch model
        model_name: Name of the model
        output_dir: Directory to save outputs
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Save the raw Fisher Information Matrix
    print(f"Saving Fisher Information Matrix to {output_dir / f'{model_name}_fisher_matrix.npy'}...")
    np.save(output_dir / f'{model_name}_fisher_matrix.npy', fisher_info)
    
    # Convert to CuPy array for GPU computation
    print("Converting to CuPy array for GPU eigenvalue decomposition...")
    fisher_info_cp = cp.array(fisher_info)
    
    # Calculate eigenvalues using CuPy (on GPU)
    print("Computing eigenvalues on GPU...")
    eigenvalues, eigenvectors = cp.linalg.eigh(fisher_info_cp)
    
    # Convert back to NumPy for further processing
    eigenvalues = cp.asnumpy(eigenvalues)
    eigenvectors = cp.asnumpy(eigenvectors)
    
    # Sort eigenvalues in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Save eigenvalues and eigenvectors
    np.save(output_dir / f'{model_name}_eigenvalues.npy', eigenvalues)
    np.save(output_dir / f'{model_name}_eigenvectors.npy', eigenvectors)
    
    # Plot eigenvalue distribution
    plt.figure(figsize=(10, 6))
    plt.hist(eigenvalues**2, bins=50)
    plt.title(f'Eigenvalue^2 Distribution of Fisher Information Matrix\n{model_name} ({num_params} parameters)')
    plt.xlabel('Eigenvalue^2')
    plt.ylabel('count')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(output_dir / f'{model_name}_fisher_eigenvalues.png')
    plt.close()
    
    # Plot CDF of eigenvalues
    plt.figure(figsize=(10, 6))
    sorted_evals = np.sort(eigenvalues**2)
    cdf = np.arange(1, len(sorted_evals) + 1) / len(sorted_evals) * 100
    plt.plot(sorted_evals, cdf)
    plt.title(f'CDF of Eigenvalue^2 Distribution\n{model_name} ({num_params} parameters)')
    plt.xlabel('Eigenvalue^2')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    plt.savefig(output_dir / f'{model_name}_fisher_eigenvalues_cdf.png')
    plt.close()
    
    # Plot complementary CDF (1-CDF) with log scale
    plt.figure(figsize=(10, 6))
    ccdf = 1 - (np.arange(1, len(sorted_evals) + 1) / len(sorted_evals))  # Complementary CDF as ratio
    plt.plot(sorted_evals, ccdf)
    plt.title(f'Complementary CDF (1-CDF) of Eigenvalue^2 Distribution\n{model_name} ({num_params} parameters)')
    plt.xlabel('Eigenvalue^2')
    plt.ylabel('Ratio')
    plt.xscale('log')  # Add log scale to x-axis
    plt.yscale('log')
    plt.grid(True, which="both")  # Show grid for both major and minor ticks
    plt.savefig(output_dir / f'{model_name}_fisher_eigenvalues_ccdf_log.png')
    plt.close()
    
    # Calculate statistics
    stats = {
        'max_eigenvalue': eigenvalues[0],
        'min_eigenvalue': eigenvalues[-1],
        'mean_eigenvalue': np.mean(eigenvalues),
        'median_eigenvalue': np.median(eigenvalues),
        'std_eigenvalue': np.std(eigenvalues),
        'condition_number': eigenvalues[0] / eigenvalues[-1],
        'effective_rank': np.sum(eigenvalues) / eigenvalues[0],
        'num_parameters': num_params
    }
    
    # Save statistics
    with open(output_dir / f'{model_name}_fisher_stats.txt', 'w') as f:
        for key, value in stats.items():
            f.write(f'{key}: {value:.6f}\n')
    
    return stats

def main(model):
    start_time = time.time()
    
    # Initialize data loader
    data_loader = MNISTDataLoader(batch_size=64, preload_gpu=True)
    train_loader = data_loader.get_train_loader()
    
    model_name = model._get_name()
    # Print model parameters and FIM size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    print(f"Fisher Information Matrix size: {num_params:,} x {num_params:,} = {num_params**2:,}")
    # Output directory
    output_dir = Path(f'model_interpretation/outputs/fisher_analysis/{model_name}/')
    
    print(f"\nAnalyzing {model_name}...")
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Calculate Fisher Information Matrix
    fim_start_time = time.time()
    fisher_info = calculate_fisher_information(model, train_loader)
    fim_end_time = time.time()
    print("Fisher Information Matrix calculated.")
    print(f"FIM calculation took {fim_end_time - fim_start_time:.2f} seconds")
    
    # Analyze and save results
    analysis_start_time = time.time()
    stats = analyze_fisher_information(fisher_info, model, model_name, output_dir)
    analysis_end_time = time.time()
    
    # Print summary statistics
    print(f"\nFisher Information Analysis for {model_name}:")
    print(f"Max eigenvalue: {stats['max_eigenvalue']:.6f}")
    print(f"Min eigenvalue: {stats['min_eigenvalue']:.6f}")
    print(f"Condition number: {stats['condition_number']:.6f}")
    print(f"Effective rank: {stats['effective_rank']:.6f}")
    print(f"Number of parameters: {stats['num_parameters']}")
    
    # Print timing information
    total_time = time.time() - start_time
    print(f"\nTiming Summary:")
    print(f"FIM calculation: {fim_end_time - fim_start_time:.2f} seconds")
    print(f"Eigenvalue analysis: {analysis_end_time - analysis_start_time:.2f} seconds")
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
        # Model to analyze
    model = StandardFullyConnected()
    main(model) 