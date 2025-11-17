import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import cupy as cp
import time
from torch.utils.data import Subset, DataLoader

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import MNISTDataLoader
from models.specific_models.ShiftInvariantCNN import ShiftInvariantCNN
from models.specific_models.StandardFullyConnected import StandardFullyConnected
from models.specific_models.LinearModel import LinearModel
from torch.autograd.functional import hessian

def calculate_fisher_information(model, data_loader, enable_diagnostics=True, prob_threshold=None):
    """
    Calculate the empirical Fisher Information Matrix using the outer product form.
    FIM = E[∇log p(y|x,θ)∇log p(y|x,θ)^T]
    
    Args:
        model: PyTorch model to analyze
        data_loader: DataLoader for the dataset  
        enable_diagnostics: If True, collect diagnostic vectors (prob_values, score_values, etc.)
        prob_threshold: If provided, skip gradient computations for samples with probability below this threshold
        
    Returns:
        fisher_info: Fisher Information Matrix (numpy array)
        diagnostics: Dictionary with diagnostic vectors (only if enable_diagnostics=True, else None)
    """
    model.eval()  # Set to eval to remove batch normalization and dropout.
    num_params = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device
    
    # Initialize FIM as zero matrix
    fisher_info = torch.zeros((num_params, num_params), device=device)
    loss_function = nn.CrossEntropyLoss()
    # Initialize diagnostics if requested
    diagnostics = None
    if enable_diagnostics:
        # Calculate total number of gradient computations
        total_computations = 0
        for batch_idx, (data, _) in enumerate(16):
            batch_size = data.size(0)
            outputs = model(data.to(device))
            num_classes = outputs.size(1)
            total_computations += batch_size * num_classes
        print(f"Pre-allocating vectors for {total_computations} gradient computations...")
        # Pre-allocate diagnostic vectors
        prob_values = np.zeros(total_computations, dtype=np.float32)
        score_values = np.zeros(total_computations, dtype=np.float32)
        fisher_matrix_norms = np.zeros(total_computations, dtype=np.float32)
        grad_norms = np.zeros(total_computations, dtype=np.float32)
    
    # Calculate FIM using gradients
    nof_samples = 0
    computation_idx = 0
    skipped_samples = 0
    
    for batch_idx, (data, targets) in enumerate(data_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        # Define a function that computes the loss for a given set of parameters
        def loss_fn(params):
            # Reshape flat parameters back to original shapes
            param_idx = 0
            param_dict = {}
            for name, param in model.named_parameters():
                param_size = param.numel()
                param_dict[name] = params[param_idx:param_idx + param_size].view(param.shape)
                param_idx += param_size
            
            # Use functional_call to run model with new parameters without modifying original
            from torch.func import functional_call
            outputs = functional_call(model, param_dict, data)
            return loss_function(outputs, targets)
        
        flat_params = torch.cat([p.view(-1) for p in model.parameters()])
        flat_params = flat_params.detach().requires_grad_(True)  # Ensure gradients are tracked
        H = hessian(loss_fn, flat_params)
        fisher_info.add_(H)
        if (batch_idx + 1) % 10 == 0:
            print(f"Analyzed {(batch_idx + 1) * data.size(0)} samples...")

    
    # Normalize by number of batches
    fisher_info /= len(data_loader)
    
    # Package diagnostic vectors if enabled
    if enable_diagnostics:
        diagnostics = {
            'prob_values': 0,
            'score_values': 0,
            'fisher_matrix_norms': 0,
            'grad_norms': 0
        }
        return fisher_info.cpu().numpy(), diagnostics
    else:
        return fisher_info.cpu().numpy()

def analyze_fisher_information(fisher_info, diagnostics, model, model_name, output_dir):
    """
    Analyze the Fisher Information Matrix through spectral decomposition.
    Optionally saves diagnostic vectors if provided.
    
    Args:
        fisher_info: The Fisher Information Matrix
        diagnostics: Dictionary containing diagnostic vectors (can be None)
        model: The PyTorch model
        model_name: Name of the model
        output_dir: Directory to save outputs
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Save diagnostic vectors if provided
    if diagnostics is not None:
        print("Saving diagnostic vectors...")
        for key, values in diagnostics.items():
            np.save(output_dir / f'{model_name}_{key}.npy', values)
            print(f"Saved {key}: shape {values.shape}, min={values.min():.6e}, max={values.max():.6e}")
    else:
        print("No diagnostic vectors to save (diagnostics disabled)")
    
    # Save the raw Fisher Information Matrix
    print(f"Saving Fisher Information Matrix to {output_dir / f'{model_name}_fisher_matrix.npy'}...")
    # Convert PyTorch tensor to NumPy array before saving
    fisher_info_np = fisher_info.cpu().numpy() if hasattr(fisher_info, 'cpu') else fisher_info
    np.save(output_dir / f'{model_name}_fisher_matrix.npy', fisher_info_np)
    
    # Convert to CuPy array for GPU computation
    print("Converting to CuPy array for GPU eigenvalue decomposition...")
    fisher_info_cp = cp.array(fisher_info_np)
    
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
    plt.hist(eigenvalues, bins=50)
    plt.title(f'Eigenvalue Distribution of Fisher Information Matrix\n{model_name} ({num_params} parameters)')
    plt.xlabel('Eigenvalue')
    plt.ylabel('count')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(output_dir / f'{model_name}_fisher_eigenvalues.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(np.sort(np.abs(eigenvalues))[::-1])
    plt.title(f'absolute Eigenvalue Spectrum of Fisher Information Matrix\n{model_name} ({num_params} parameters)')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(output_dir / f'{model_name}_fisher_eigenvalues_abs.png')
    plt.close()
    
    

    
    # plot eigenvalues dist
    
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

def main(model, model_name, data_loader, enable_diagnostics=True, prob_threshold=None, num_train_samples=None):
    """
    Main function to calculate and analyze Fisher Information Matrix.
    
    Args:
        model: PyTorch model to analyze
        model_name: Name of the model for output files
        data_loader: DataLoader for the dataset
        enable_diagnostics: Whether to collect diagnostic information
        prob_threshold: Optional probability threshold for sample filtering
        num_train_samples: Optional limit on the number of training samples to use
    """
    start_time = time.time()
    train_loader = data_loader.get_train_loader()
    if num_train_samples is not None:
        base_dataset = train_loader.dataset
        limit = min(num_train_samples, len(base_dataset))
        subset_indices = list(range(limit))
        subset = Subset(base_dataset, subset_indices)
        train_loader = DataLoader(
            subset,
            batch_size=train_loader.batch_size,
            shuffle=False,
            pin_memory=getattr(train_loader, 'pin_memory', True),
            num_workers=getattr(train_loader, 'num_workers', 0)
        )
    # Print model parameters and FIM size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    print(f"Fisher Information Matrix size: {num_params:,} x {num_params:,} = {num_params**2:,}")
    # Output directory
    output_dir = Path(f'model_interpretation/outputs/fisher_analysis_hessian/{model_name}/')
    
    print(f"\nAnalyzing {model_name}...")
    if enable_diagnostics:
        print("Diagnostics enabled - collecting detailed sample information")
    else:
        print("Diagnostics disabled - faster computation")
    if prob_threshold is not None:
        print(f"Using probability threshold: {prob_threshold}")
    if num_train_samples is not None:
        print(f"Limiting Fisher/Hessian computation to first {num_train_samples} training samples")
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Calculate Fisher Information Matrix
    fim_start_time = time.time()
    result = calculate_fisher_information(model, train_loader, enable_diagnostics, prob_threshold)
    if enable_diagnostics:
        fisher_info, diagnostics = result
    else:
        fisher_info, diagnostics = result, None
    fim_end_time = time.time()
    print("Fisher Information Matrix calculated.")
    print(f"FIM calculation took {fim_end_time - fim_start_time:.2f} seconds")
    
    # Analyze and save results
    analysis_start_time = time.time()
    stats = analyze_fisher_information(fisher_info, diagnostics, model, model_name, output_dir)
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
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Use the improved interactive model loader
    from utils.model_loader import load_model_interactive
    
    # Let the user choose which model and checkpoint to analyze
    print("\nSelect a model and checkpoint for Fisher Information Analysis")
    model, model_name, data_loader = load_model_interactive()
    
    # Ask user about options
    enable_diag = input("\nEnable diagnostics collection? (y/n, default=y): ").lower()
    enable_diagnostics = enable_diag != 'n'
    
    prob_thresh = input("Enter probability threshold (or press Enter for none): ").strip()
    prob_threshold = float(prob_thresh) if prob_thresh else None
    
    # Ask for number of training samples to use (optional)
    num_samples_input = input("Enter number of training samples to use (or press Enter for all): ").strip()
    num_train_samples = int(num_samples_input) if num_samples_input else None
    if num_train_samples is not None and num_train_samples <= 0:
        print("Non-positive sample count provided; using all samples.")
        num_train_samples = None
    
    # Run the main function with the selected model
    main(model, model_name, data_loader, enable_diagnostics, prob_threshold, num_train_samples)