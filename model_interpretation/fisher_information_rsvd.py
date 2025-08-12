import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import cupy as cp
import time
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def apply_fisher_to_matrix(model, data_loader, Q, device, side='right'):
    """
    Implicitly apply the Fisher Information Matrix to matrix Q.
    
    Args:
        model: The PyTorch model
        data_loader: Data loader for the dataset
        Q: Matrix to apply Fisher to (num_params x k)
        device: Device to use for computation
        side: 'right' or 'left' (left will use Q.T)
        
    Returns:
        Result of Fisher Information Matrix applied to X
    """
    num_params = sum(p.numel() for p in model.parameters())
    k = Q.size(1)
    
    # Initialize result matrix Y = Q.T @ A
    if side == 'right':
        fisher_info_projection = torch.zeros((num_params, k), device=device)
    if side == 'left':
        fisher_info_projection = torch.zeros((k, num_params), device=device)
    error = 0
    # Keep track of samples processed
    total_samples = 0
    
    # Process batches
    for batch_idx, (data, _) in enumerate(tqdm(data_loader, desc="Computing fisher information matrix projection")):
        batch_size = data.size(0)
        total_samples += batch_size
        data = data.to(device)
        
        # Forward pass
        outputs = model(data)
        probs = torch.softmax(outputs, dim=1)
        log_probs = torch.log(probs)
        
        # Process each sample in the batch
        for i in range(data.size(0)):
            # Compute gradients for each class
            for j in range(probs.size(1)):
                # Get score for this sample and class
                score = log_probs[i, j].clone()
                prob = probs[i, j].item()  # Get scalar value
                
                
                # Compute gradient with respect to parameters
                model.zero_grad()
                score.backward(retain_graph=True)
                
                # Get flattened gradient and detach
                grad = torch.cat([p.grad.detach().view(-1) for p in model.parameters()])
                grad = grad.view(num_params, 1)

                # Add outer product to Fisher Information Matrix
                if prob > 0:
                    if side == 'right':
                        proj = grad.T @ Q # grad @ Q => (1, n) @ (n, k) => (1, k) where n is params, k is projection dim
                        fisher_info_projection.add_(prob * grad @ proj) # (n, 1) @ (1, k) => (n, k)
                        error += prob * (grad.norm(2) - proj.norm(2))
                    if side =='left':
                        proj = Q.T @ grad # Q.T @ grad => (k, n) @ (n, 1) => (k, 1) where n is params, k is projection dim
                        fisher_info_projection.add_(prob * proj @ grad.T) # (k, 1) @ (1, n) => (k, n)
                        error += prob * (grad.norm(2) - proj.norm(2))

                    

                
    # Average over total samples
    fisher_info_projection /= total_samples
    error = error / total_samples
    return fisher_info_projection, error


def calculate_fisher_rsvd(model, data_loader, k, power_iterations=1):
    """
    Calculate the Fisher Information Matrix using Randomized SVD (RSI algorithm).
    
    Args:
        model: The PyTorch model
        data_loader: Data loader for the dataset
        k: Number of random projections/dimensions to use
        power_iterations: Number of power iterations to enhance accuracy (default: 1)
        
    Returns:
        U, Sigma, V: The SVD components of the approximated Fisher Information Matrix
    """
    model.train()  # Set to training mode for gradient computation
    num_params = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device
    
    print(f"RSVD parameters: k={k}, power_iterations={power_iterations}")
    
    # Step 1: Generate a random matrix X ∈ ℝⁿᵏ
    print("Step 1: Generating random matrix X")
    torch.manual_seed(42)  # For reproducibility
    X = torch.randn(num_params, k, device=device)
    
    # Step 2: For i = 0, ..., q, calculate the QR factorization AX = QR and update X = A*Q
    for i in range(power_iterations):
        print(f"Step 2.{i}: range finder: power iteration {i+1}/{power_iterations}")
        
        # NOTE: this is not the proper way to do it. need  normalization to avoid rounding errors, see algorithm 4.4 at Halko.
        # this is OK for single power itterations (no power iterations)
        # Apply Fisher Information Matrix to X
        AX, e = apply_fisher_to_matrix(model, data_loader, X, device, side='right')

        
    # Step 3: QR factorization
    print("Step 3: QR decomposition (Q = estimated range basis)")
    Q, R = torch.linalg.qr(AX)
    
    # Step 4: project onto estimated range
    print("Step 4: project into estimated Q range")
    B, error = apply_fisher_to_matrix(model, data_loader, Q, device, side='left')
    print("Error term = ", error)
    # Step 5: Calculate the SVD X = VΣU*
    print("Step 5: Calculating SVD of final matrix")
    # Note: In PyTorch, SVD returns U, S, V where X = USV^T
    U_hat, S, V_hat = torch.linalg.svd(B, full_matrices=False)
    
    # Step 6: Set U = QÛ
    print("Step 6: Computing Eigenvectors U = QU' (U' is the final matrix eigenvectors)")
    U = Q @ U_hat
    
    # Return the approximation components - U is the eigenvectors, S^2 are eigenvalues
    return U.cpu().numpy(), S.cpu().numpy(), V_hat.cpu().numpy(), error


def analyze_fisher_information_rsvd(U, S, V, model, model_name, output_dir, error=None):
    """
    Analyze the Fisher Information Matrix through its SVD decomposition.
    
    Args:
        U, S, V: SVD components from RSVD
        model: The PyTorch model
        model_name: Name of the model
        output_dir: Directory to save outputs
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Save SVD components
    np.save(output_dir / f'{model_name}_U_rsvd.npy', U)
    np.save(output_dir / f'{model_name}_S_rsvd.npy', S)
    np.save(output_dir / f'{model_name}_V_rsvd.npy', V)
    
    # The eigenvalues of the Fisher Information Matrix are the squares of singular values
    eigenvalues = S
    
    # Plot eigenvalue distribution
    plt.figure(figsize=(10, 6))
    plt.hist(eigenvalues, bins=50)
    plt.title(f'Eigenvalue Distribution of Fisher Information Matrix (RSVD)\n{model_name} ({num_params} parameters)')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(output_dir / f'{model_name}_fisher_eigenvalues_rsvd.png')
    plt.close()
    
    # Plot CDF of eigenvalues
    plt.figure(figsize=(10, 6))
    sorted_evals = np.sort(eigenvalues)
    cdf = np.arange(1, len(sorted_evals) + 1) / len(sorted_evals) * 100
    plt.plot(sorted_evals, cdf)
    plt.title(f'CDF of Eigenvalue Distribution (RSVD)\n{model_name} ({num_params} parameters)')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    plt.savefig(output_dir / f'{model_name}_fisher_eigenvalues_cdf_rsvd.png')
    plt.close()
    
    # Plot complementary CDF (1-CDF) with log scale
    plt.figure(figsize=(10, 6))
    ccdf = 1 - (np.arange(1, len(sorted_evals) + 1) / len(sorted_evals))  # Complementary CDF as ratio
    plt.plot(sorted_evals, ccdf)
    plt.title(f'Complementary CDF (1-CDF) of Eigenvalue Distribution (RSVD)\n{model_name} ({num_params} parameters)')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Ratio')
    plt.xscale('log')  # Add log scale to x-axis
    plt.yscale('log')
    plt.grid(True, which="both")  # Show grid for both major and minor ticks
    plt.savefig(output_dir / f'{model_name}_fisher_eigenvalues_ccdf_log_rsvd.png')
    plt.close()
    
    # Calculate statistics
    stats = {
        'max_eigenvalue': eigenvalues[0],
        'min_eigenvalue': eigenvalues[-1],
        'mean_eigenvalue': np.mean(eigenvalues),
        'median_eigenvalue': np.median(eigenvalues),
        'std_eigenvalue': np.std(eigenvalues),
        'condition_number': eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else float('inf'),
        'effective_rank': np.sum(eigenvalues) / eigenvalues[0],
        'num_parameters': num_params
    }
    
    # Add error information if provided
    if error is not None:
        stats['rsvd_error_bound'] = error
    
    # Save statistics
    with open(output_dir / f'{model_name}_fisher_stats_rsvd.txt', 'w') as f:
        for key, value in stats.items():
            f.write(f'{key}: {value:.6f}\n')
    
    return stats


def main(model, model_name, data_loader):
    start_time = time.time()
    train_loader = data_loader.get_train_loader()
    
    # Store the original model name for directory structure
    original_model_name = model_name
    
    # Get model class name for easier reference in output files
    model_class_name = model._get_name()
    
    # Print model parameters and FIM size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    print(f"Fisher Information Matrix size: {num_params:,} x {num_params:,} = {num_params**2:,}")
    
    # Output directory - use the original model name from training
    output_dir = Path(f'model_interpretation/outputs/fisher_analysis/{original_model_name}/')
    
    print(f"\nAnalyzing {original_model_name} (model type: {model_class_name})...")
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Calculate Fisher Information Matrix using RSVD
    fim_start_time = time.time()
    k = 1000  # Number of components to extract
    power_iterations = 1  # Number of power iterations
    U, S, V, error = calculate_fisher_rsvd(model, train_loader, k, power_iterations)
    fim_end_time = time.time()
    print("Fisher Information Matrix analysis with RSVD completed.")
    print(f"RSVD calculation took {fim_end_time - fim_start_time:.2f} seconds")
    
    # Analyze and save results
    analysis_start_time = time.time()
    stats = analyze_fisher_information_rsvd(U, S, V, model, original_model_name, output_dir, error)
    analysis_end_time = time.time()
    
    # Print summary statistics
    print(f"\nFisher Information Analysis for {original_model_name} (RSVD):")
    print(f"Max eigenvalue: {stats['max_eigenvalue']:.6f}")
    print(f"Min eigenvalue: {stats['min_eigenvalue']:.6f}")
    print(f"Condition number: {stats['condition_number']:.6f}")
    print(f"Effective rank: {stats['effective_rank']:.6f}")
    print(f"Number of parameters: {stats['num_parameters']}")
    
    # Print timing information
    total_time = time.time() - start_time
    print(f"\nTiming Summary:")
    print(f"RSVD calculation: {fim_end_time - fim_start_time:.2f} seconds")
    print(f"Analysis: {analysis_end_time - analysis_start_time:.2f} seconds")
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Use the interactive model loader
    from utils.model_loader import load_model_interactive
    
    # Let the user choose which model to analyze
    model, model_name, data_loader = load_model_interactive()
    
    # Run the main function with the selected model
    main(model, model_name, data_loader) 