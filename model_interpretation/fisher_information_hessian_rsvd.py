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
from torch.autograd.functional import hvp
from torch.utils.data import Subset, DataLoader

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def apply_fisher_to_matrix(model, data_loader, Q, device, side='right', compute_trace=False, use_labels=True):
    """
    Implicitly apply the Fisher Information Matrix to matrix Q.
    
    Args:
        model: The PyTorch model
        data_loader: Data loader for the dataset
        Q: Matrix to apply Fisher to (num_params x k)
        device: Device to use for computation
        side: 'right' or 'left' (left will use Q.T)
        compute_trace: If True, also compute the trace of Fisher Information Matrix
        use_labels: If True, use the labels to compute the Fisher Information Matrix (NLL between predicted and true labels
        If False, use the predicted probabilites as true labels for the loss function (NLL between predicted and predicted labels).
    Returns:
        Result of Fisher Information Matrix applied to X, error, and optionally trace
    """
    model.eval()  # Set to eval to remove batch normalization and dropout.
    num_params = sum(p.numel() for p in model.parameters())
    k = Q.size(1)
    loss_function = nn.CrossEntropyLoss()
    # Always compute right multiplication (A @ Q), shape (P, k)
    right_product = torch.zeros((num_params, k), device=device)
    error = torch.tensor(0.0, device=device)
    fisher_trace = torch.tensor(0.0, device=device) if compute_trace else None
    # Keep track of samples processed for proper averaging
    total_samples = 0
    nans_counter = 0

    Q = Q.to(device)

    # We will estimate trace(A) using the provided Q columns: tr(A) ≈ (1/k) * tr(Q^T (A Q))

    for batch_idx, (data, targets) in enumerate(data_loader):
        data = data.to(device)
        targets = targets.to(device)

        batch_size = data.size(0)

        # Define loss as function of flattened parameters for current batch
        def loss_fn(params):
            # Reshape flat parameters back to original shapes
            param_idx = 0
            param_dict = {}
            for name, param in model.named_parameters():
                param_size = param.numel()
                param_dict[name] = params[param_idx:param_idx + param_size].view(param.shape)
                param_idx += param_size
            # Run model with functional_call using these params
            from torch.func import functional_call
            outputs = functional_call(model, param_dict, data)
            if use_labels:
                return loss_function(outputs, targets)
            else:
                # Technical detail of cross entropy implementation.
                # the targets should be probabilities of the classes (not raw logits).
                outputs_target = torch.softmax(outputs, dim=1)
                return loss_function(outputs, outputs_target)


        # Flat parameter vector (requires grad for second-order)
        flat_params = torch.cat([p.view(-1) for p in model.parameters()])
        flat_params = flat_params.detach().to(device).requires_grad_(True)

        # Apply A (Hessian/Fisher) to each column of Q via HVP; weight by batch size
        for j in range(k):
            v = Q[:, j]
            f_output, Hv_j = hvp(loss_fn, flat_params, v)
            right_product[:, j] += Hv_j

        total_samples += batch_size

        if (batch_idx + 1) % 10 == 0:
            print(f"Analyzed {(batch_idx + 1) * data.size(0)} samples...")

    # Convert to dataset-mean (sample-weighted)
    if total_samples > 0:
        right_product /= len(data_loader) # normalize by number of batches
        if compute_trace:
            # Use existing Q to form Q^T (A Q) and take mean of diagonal as estimator
            fisher_trace = torch.trace(Q.T @ right_product) / float(k)
                    

    # Print number of samples with NaNs and number of classes (robust detection)
    num_outputs = None
    for m in reversed(list(model.modules())):
        if hasattr(m, 'out_features'):
            num_outputs = m.out_features
            break
    if num_outputs is None:
        try:
            with torch.no_grad():
                sample_data, _ = next(iter(data_loader))
                sample_out = model(sample_data.to(device))
                num_outputs = sample_out.size(1)
        except Exception:
            num_outputs = 0
    print(f"Number of samples with nans: {nans_counter} / {total_samples * num_outputs}")

    # Return according to requested side by transposing the right product if needed
    if side == 'left':
        result = right_product.T  # (k, P) = (A @ Q)^T = Q^T @ A (since A is symmetric)
    else:
        result = right_product

    if compute_trace:
        return result, error, fisher_trace.to('cpu')
    else:
        return result, error




def calculate_fisher_rsvd(model, data_loader, k, power_iterations=1, save_intermediates=True, cache_dir="rsvd_cache", use_labels=True):
    """
    Calculate the Fisher Information Matrix using Randomized SVD (RSI algorithm).
    
    Args:
        model: The PyTorch model
        data_loader: Data loader for the dataset
        k: Number of random projections/dimensions to use
        power_iterations: Number of power iterations to enhance accuracy (default: 1)
        save_intermediates: Whether to save intermediate results
        cache_dir: Directory to save intermediate results
        use_labels: If True, use ground-truth labels; otherwise use predicted probabilities
        
    Returns:
        U, Sigma, V: The SVD components of the approximated Fisher Information Matrix, error, trace, timings
    """
    model.train()  # Set to training mode for gradient computation
    num_params = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device
    
    print(f"RSVD parameters: k={k}, power_iterations={power_iterations}")
    
    # Initialize timing dictionary
    timings = {}
    
    # Create cache directory if saving intermediates
    if save_intermediates:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        print(f"Intermediate results will be saved to: {cache_path}")
    
    # Step 1: Generate a random matrix X ∈ ℝⁿᵏ
    step1_start = time.time()
    X_file = cache_path / "X_initial.npy" if save_intermediates else None
    if save_intermediates and X_file.exists():
        print("Step 1: Loading existing random matrix X")
        X = torch.from_numpy(np.load(X_file)).to(device)
        timings['step1_random_matrix'] = 0.0  # Loaded from cache
    else:
        print("Step 1: Generating random matrix X")
        torch.manual_seed(42)  # For reproducibility
        X = torch.randn(num_params, k, device=device)
        if save_intermediates:
            np.save(X_file, X.cpu().numpy())
            print(f"Saved X to {X_file}")
        timings['step1_random_matrix'] = time.time() - step1_start
    
    # Step 2: For i = 0, ..., q, calculate the QR factorization AX = QR and update X = A*Q
    step2_start = time.time()
    timings['step2_power_iterations'] = []
    AX = X  # Initialize for the case where power_iterations = 0
    for i in range(power_iterations):
        print(f"Step 2.{i}: range finder: power iteration {i+1}/{power_iterations}")
        iteration_start = time.time()
        
        # Check if AX for this iteration already exists
        AX_file = cache_path / f"AX_iteration_{i}.npy" if save_intermediates else None
        if save_intermediates and AX_file.exists():
            print(f"Loading existing AX for iteration {i}")
            AX = torch.from_numpy(np.load(AX_file)).to(device)
            iteration_time = 0.0  # Loaded from cache
        else:
            # NOTE: this is not the proper way to do it. need  normalization to avoid rounding errors, see algorithm 4.4 at Halko.
            # this is OK for single power itterations (no power iterations)
            # Apply Fisher Information Matrix to X
            AX, e = apply_fisher_to_matrix(model, data_loader, X, device, side='right', use_labels=use_labels)
            if save_intermediates:
                np.save(AX_file, AX.cpu().numpy())
                print(f"Saved AX iteration {i} to {AX_file}")
            iteration_time = time.time() - iteration_start
        
        timings['step2_power_iterations'].append(iteration_time)
        if iteration_time > 0:
            print(f"Power iteration {i} took: {iteration_time:.2f} seconds")
        
        # Update X for next iteration (if there is one)
        if i < power_iterations - 1:
            X = AX
    
    timings['step2_total'] = time.time() - step2_start

    # Step 3: QR factorization
    step3_start = time.time()
    Q_file = cache_path / "Q_matrix.npy" if save_intermediates else None
    R_file = cache_path / "R_matrix.npy" if save_intermediates else None
    
    if save_intermediates and Q_file.exists() and R_file.exists():
        print("Step 3: Loading existing QR decomposition")
        Q = torch.from_numpy(np.load(Q_file)).to(device)
        R = torch.from_numpy(np.load(R_file)).to(device)
        timings['step3_qr_decomposition'] = 0.0  # Loaded from cache
    else:
        print("Step 3: QR decomposition (Q = estimated range basis)")
        Q, R = torch.linalg.qr(AX)
        if save_intermediates:
            np.save(Q_file, Q.cpu().numpy())
            np.save(R_file, R.cpu().numpy())
            print(f"Saved Q to {Q_file}")
            print(f"Saved R to {R_file}")
        timings['step3_qr_decomposition'] = time.time() - step3_start
    
    if timings['step3_qr_decomposition'] > 0:
        print(f"QR decomposition took: {timings['step3_qr_decomposition']:.2f} seconds")
    
    # Step 4: project onto estimated range
    step4_start = time.time()
    B_file = cache_path / "B_matrix.npy" if save_intermediates else None
    error_file = cache_path / "B_error.npy" if save_intermediates else None
    fisher_trace_file = cache_path / "fisher_trace.npy" if save_intermediates else None
    if save_intermediates and B_file.exists() and error_file.exists():
        print("Step 4: Loading existing B matrix and error")
        B = torch.from_numpy(np.load(B_file)).to(device)
        fisher_trace = np.load(fisher_trace_file).item()
        error = np.load(error_file).item()
        timings['step4_projection'] = 0.0  # Loaded from cache
    else:
        print("Step 4: project into estimated Q range")
        B, error, fisher_trace = apply_fisher_to_matrix(model, data_loader, Q, device, side='left', compute_trace=True, use_labels=use_labels)
        if save_intermediates:
            np.save(B_file, B.cpu().numpy())
            np.save(error_file, np.array(error.cpu().numpy()))
            # Save fisher_trace as a plain Python float to avoid NumPy deprecation warnings
            np.save(fisher_trace_file, float(fisher_trace))
            print(f"Saved B to {B_file}")
            print(f"Saved error to {error_file} (will also be saved in final statistics)")
            print(f"Saved fisher trace to {fisher_trace_file}")
        timings['step4_projection'] = time.time() - step4_start
    
    print("Error term = ", error)
    if timings['step4_projection'] > 0:
        print(f"Projection step took: {timings['step4_projection']:.2f} seconds")
    
    print("Fisher trace = ", fisher_trace)
    
    # Step 5: Calculate the SVD X = VΣU*
    step5_start = time.time()
    U_hat_file = cache_path / "U_hat.npy" if save_intermediates else None
    S_file = cache_path / "S_singular_values.npy" if save_intermediates else None
    V_hat_file = cache_path / "V_hat.npy" if save_intermediates else None
    
    if save_intermediates and U_hat_file.exists() and S_file.exists() and V_hat_file.exists():
        print("Step 5: Loading existing SVD components")
        U_hat = torch.from_numpy(np.load(U_hat_file)).to(device)
        S = torch.from_numpy(np.load(S_file)).to(device)
        V_hat = torch.from_numpy(np.load(V_hat_file)).to(device)
        timings['step5_svd'] = 0.0  # Loaded from cache
    else:
        print("Step 5: Calculating SVD of final matrix")
        # Note: In PyTorch, SVD returns U, S, V where X = USV^T
        U_hat, S, V_hat = torch.linalg.svd(B, full_matrices=False)
        if save_intermediates:
            np.save(U_hat_file, U_hat.cpu().numpy())
            np.save(S_file, S.cpu().numpy())
            np.save(V_hat_file, V_hat.cpu().numpy())
            print(f"Saved U_hat to {U_hat_file}")
            print(f"Saved S to {S_file}")
            print(f"Saved V_hat to {V_hat_file}")
        timings['step5_svd'] = time.time() - step5_start
    
    if timings['step5_svd'] > 0:
        print(f"SVD computation took: {timings['step5_svd']:.2f} seconds")
    
    # Step 6: Set U = QÛ
    step6_start = time.time()
    U_final_file = cache_path / "U_final.npy" if save_intermediates else None
    print("B eigenvalues sum = ", S.sum())

    if save_intermediates and U_final_file.exists():
        print("Step 6: Loading existing final eigenvectors U")
        U = torch.from_numpy(np.load(U_final_file)).to(device)
        timings['step6_final_eigenvectors'] = 0.0  # Loaded from cache
    else:
        print("Step 6: Computing Eigenvectors U = QU' (U' is the final matrix eigenvectors)")
        U = Q @ U_hat
        if save_intermediates:
            np.save(U_final_file, U.cpu().numpy())
            print(f"Saved final U to {U_final_file}")
        timings['step6_final_eigenvectors'] = time.time() - step6_start
    
    if timings['step6_final_eigenvectors'] > 0:
        print(f"Final eigenvector computation took: {timings['step6_final_eigenvectors']:.2f} seconds")
    
    # Return the approximation components - U is the eigenvectors, S^2 are eigenvalues
    return U.cpu().numpy(), S.cpu().numpy(), V_hat.cpu().numpy(), error, fisher_trace, timings


def analyze_fisher_information_rsvd(U, S, V, model, model_name, output_dir, error=None, fisher_trace=None, timings=None, data_type='train'):
    """
    Analyze the Fisher Information Matrix through its SVD decomposition.
    
    Args:
        U, S, V: SVD components from RSVD
        model: The PyTorch model
        model_name: Name of the model
        output_dir: Directory to save outputs
        error: Error bound from RSVD computation
        fisher_trace: Trace of the Fisher Information Matrix
        timings: Dictionary of timing information for each step
        data_type: Type of data used ('train' or 'test')
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Save SVD components
    np.save(output_dir / f'{data_type}_{model_name}_U_rsvd.npy', U)
    np.save(output_dir / f'{data_type}_{model_name}_S_rsvd.npy', S)
    np.save(output_dir / f'{data_type}_{model_name}_V_rsvd.npy', V)
    
    # The eigenvalues of the Fisher Information Matrix are the squares of singular values
    eigenvalues = S
    
    # Plot eigenvalue distribution
    plt.figure(figsize=(10, 6))
    plt.hist(eigenvalues, bins=50)
    plt.title(f'Eigenvalue Distribution of Fisher Information Matrix (RSVD) - {data_type.title()} Data\n{model_name} ({num_params} parameters)')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(output_dir / f'{data_type}_{model_name}_fisher_eigenvalues_rsvd.png')
    plt.close()
    
    # Plot CDF of eigenvalues
    plt.figure(figsize=(10, 6))
    sorted_evals = np.sort(eigenvalues)
    cdf = np.arange(1, len(sorted_evals) + 1) / len(sorted_evals) * 100
    plt.plot(sorted_evals, cdf)
    plt.title(f'CDF of Eigenvalue Distribution (RSVD) - {data_type.title()} Data\n{model_name} ({num_params} parameters)')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    plt.savefig(output_dir / f'{data_type}_{model_name}_fisher_eigenvalues_cdf_rsvd.png')
    plt.close()
    
    # Plot complementary CDF (1-CDF) with log scale
    plt.figure(figsize=(10, 6))
    ccdf = 1 - (np.arange(1, len(sorted_evals) + 1) / len(sorted_evals))  # Complementary CDF as ratio
    plt.plot(sorted_evals, ccdf)
    plt.title(f'Complementary CDF (1-CDF) of Eigenvalue Distribution (RSVD) - {data_type.title()} Data\n{model_name} ({num_params} parameters)')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Ratio')
    plt.xscale('log')  # Add log scale to x-axis
    plt.yscale('log')
    plt.grid(True, which="both")  # Show grid for both major and minor ticks
    plt.savefig(output_dir / f'{data_type}_{model_name}_fisher_eigenvalues_ccdf_log_rsvd.png')
    plt.close()
    
    # Plot simple eigenvalue plot (index vs value)
    plt.figure(figsize=(10, 6))
    eigenvalue_indices = np.arange(1, len(eigenvalues) + 1)
    plt.plot(eigenvalue_indices, eigenvalues, 'b-', linewidth=1.5)
    plt.title(f'Eigenvalue Spectrum (RSVD) - {data_type.title()} Data\n{model_name} ({num_params} parameters)')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'{data_type}_{model_name}_fisher_eigenvalue_spectrum_rsvd.png')
    plt.close()
    
    # Calculate statistics
    fisher_trace_val = fisher_trace if fisher_trace is not None else 0.0
    rsvd_trace_val = np.sum(eigenvalues)
    
    stats = {
        'fisher_trace': fisher_trace_val,
        'rsvd_trace_approximation': rsvd_trace_val,  # Sum of all eigenvalues from RSVD
        'trace_approximation_ratio': rsvd_trace_val / fisher_trace_val if fisher_trace_val > 0 else 0.0,
        'max_eigenvalue': eigenvalues[0],
        'min_eigenvalue': eigenvalues[-1],
        'mean_eigenvalue': np.mean(eigenvalues),
        'median_eigenvalue': np.median(eigenvalues),
        'std_eigenvalue': np.std(eigenvalues),
        'condition_number': eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else float('inf'),
        'effective_rank': np.sum(eigenvalues) / eigenvalues[0],
        'num_parameters': num_params,
        'rsvd_error_bound': error if error is not None else 0.0,
        'num_components_k': len(eigenvalues)
    }
    
    # Add timing information if provided
    if timings is not None:
        stats['timing_trace_computation'] = timings.get('trace_computation', 0.0)
        stats['timing_step1_random_matrix'] = timings.get('step1_random_matrix', 0.0)
        stats['timing_step2_total'] = timings.get('step2_total', 0.0)
        stats['timing_step3_qr_decomposition'] = timings.get('step3_qr_decomposition', 0.0)
        stats['timing_step4_projection'] = timings.get('step4_projection', 0.0)
        stats['timing_step5_svd'] = timings.get('step5_svd', 0.0)
        stats['timing_step6_final_eigenvectors'] = timings.get('step6_final_eigenvectors', 0.0)
        # Add individual power iteration timings if available
        if 'step2_power_iterations' in timings:
            for i, iteration_time in enumerate(timings['step2_power_iterations']):
                stats[f'timing_step2_iteration_{i}'] = iteration_time
    
    # Save statistics
    with open(output_dir / f'{data_type}_{model_name}_fisher_stats_rsvd.txt', 'w') as f:
        for key, value in stats.items():
            f.write(f'{key}: {value:.6f}\n')
    
    return stats


def main(model, model_name, data_loader, num_samples=None, data_choice=None, k=None, power_iterations=1, save_intermediates=True, compute_stats=True, cache_dir_root=None, use_labels=None, random_weights=False, weights_seed=None):
    start_time = time.time()
    
    # Let user choose between train and test data
    if data_choice is None:
        print("\nChoose data for Fisher Information analysis:")
        print("1. Train data")
        print("2. Test data")
        
        while True:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == '1':
                data_for_analysis = data_loader.get_train_loader()
                data_type = 'train'
                break
            elif choice == '2':
                data_for_analysis = data_loader.get_test_loader()
                data_type = 'test'
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    else:
        if data_choice == 'train':
            data_for_analysis = data_loader.get_train_loader()
            data_type = 'train'
        elif data_choice == 'test':
            data_for_analysis = data_loader.get_test_loader()
            data_type = 'test'
        else:
            raise ValueError("data_choice must be 'train' or 'test'")
    
    print(f"Using {data_type} data for Fisher Information analysis.")

    # Optionally limit number of samples by creating a Subset-backed DataLoader
    if num_samples is not None:
        base_dataset = data_for_analysis.dataset
        limit = min(num_samples, len(base_dataset))
        subset_indices = list(range(limit))
        subset = Subset(base_dataset, subset_indices)
        data_for_analysis = DataLoader(
            subset,
            batch_size=data_for_analysis.batch_size,
            shuffle=False,
            pin_memory=getattr(data_for_analysis, 'pin_memory', True),
            num_workers=getattr(data_for_analysis, 'num_workers', 0)
        )
        print(f"Limiting RSVD computation to first {limit} {data_type} samples")
    
    # Ask interactively about using labels if not provided via CLI
    if use_labels is None:
        ul = input("\nUse ground-truth labels for loss? (y/n, default=y): ").strip().lower()
        use_labels = (ul != 'n')
    
    # Let user choose k (number of components)
    if k is None:
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("Choose the number of components (k) for RSVD:")
        print("Recommended values:")
        print("- Small models (<100k params): 100-500")
        print("- Medium models (100k-1M params): 500-1000")
        print("- Large models (>1M params): 1000-2000")
        
        while True:
            try:
                k_val = int(input("Enter k (number of components): ").strip())
                if k_val <= 0:
                    print("k must be a positive integer.")
                    continue
                num_params = sum(p.numel() for p in model.parameters())
                if k_val >= num_params:
                    print(f"k ({k_val}) must be less than the number of parameters ({num_params}).")
                    continue
                k = k_val
                break
            except ValueError:
                print("Please enter a valid integer.")
    else:
        num_params = sum(p.numel() for p in model.parameters())
        if k <= 0 or k >= num_params:
            raise ValueError(f"k must be in (0, {num_params})")
    
    print(f"Using k = {k} components for RSVD.")
    
    # Store the original model name for directory structure
    original_model_name = model_name
    
    # Get model class name for easier reference in output files
    model_class_name = model._get_name()
    
    # Print Fisher Information Matrix size (in bytes, assuming float32 = 4 bytes)
    num_params = sum(p.numel() for p in model.parameters())
    fim_bytes = num_params**2 * 4
    rsvd_bytes = num_params * k * 4
    print(f"Fisher Information Matrix size: {num_params:,} x {num_params:,} = {num_params**2:,} elements ({fim_bytes:,} bytes)")
    print(f"RSVD matrix (B) size: {num_params} x {k} = {num_params*k:,} elements ({rsvd_bytes:,} bytes)")
    
    # Output directory - use the original model name from training
    seed_suffix = f"_random_weights_seed_{weights_seed}" if (random_weights and (weights_seed is not None)) else ""
    output_dir = Path(f'model_interpretation/outputs/fisher_analysis_hessian/{data_type}_{original_model_name}{seed_suffix}/')
    # Ensure output directory exists even when skipping analysis/stats
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalyzing {original_model_name} (model type: {model_class_name})...")
    print(f"use_labels for loss computation: {use_labels}")
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Calculate Fisher Information Matrix using RSVD
    fim_start_time = time.time()
    power_iterations = int(power_iterations) if power_iterations is not None else 1
    
    # Create cache directory specific to this model and data type
    cache_root = cache_dir_root or "model_interpretation/outputs/fisher_analysis_hessian/rsvd_cache"
    cache_dir = f"{cache_root}/{data_type}_{original_model_name}{seed_suffix}_k{k}_p{power_iterations}"
    
    # Print a clean configuration summary before starting the run
    print("\nConfiguration:")
    print(f"  Model name: {original_model_name}")
    print(f"  Model type: {model_class_name}")
    print(f"  Data split: {data_type}")
    print(f"  k (components): {k}")
    print(f"  Power iterations: {power_iterations}")
    print(f"  Use labels: {use_labels}")
    print(f"  Random weights: {bool(random_weights)}")
    print(f"  Weights seed: {weights_seed if (random_weights and (weights_seed is not None)) else 'None'}")
    print(f"  Sample limit: {num_samples if 'num_samples' in locals() else 'N/A'}")
    print(f"  Save intermediates: {bool(save_intermediates)}")
    print(f"  Compute stats/plots: {bool(compute_stats)}")
    print(f"  Output dir: {output_dir}")
    print(f"  Cache dir: {cache_dir}")
    
    U, S, V, error, fisher_trace, timings = calculate_fisher_rsvd(
        model,
        data_for_analysis,
        k,
        power_iterations,
        save_intermediates=bool(save_intermediates),
        cache_dir=cache_dir,
        use_labels=use_labels
    )
    fim_end_time = time.time()
    print("Fisher Information Matrix analysis with RSVD completed.")
    total_rsvd_time = fim_end_time - fim_start_time
    print(f"RSVD calculation took {total_rsvd_time:.2f} seconds")
    
    # Add total RSVD time to timings
    timings['total_rsvd_time'] = total_rsvd_time
    
    # Analyze and save results (optional)
    if compute_stats:
        analysis_start_time = time.time()
        stats = analyze_fisher_information_rsvd(U, S, V, model, original_model_name, output_dir, error, fisher_trace, timings, data_type)
        analysis_end_time = time.time()
        analysis_time = analysis_end_time - analysis_start_time
        stats['timing_analysis'] = analysis_time
    else:
        stats = {
            'fisher_trace': float(fisher_trace) if fisher_trace is not None else 0.0,
            'num_parameters': num_params,
            'num_components_k': len(S)
        }
    
    # Re-save statistics with updated timing information
    with open(output_dir / f'{data_type}_{original_model_name}_fisher_stats_rsvd.txt', 'w') as f:
        for key, value in stats.items():
            f.write(f'{key}: {value:.6f}\n')
    
    # Print summary statistics
    print(f"\nFisher Information Analysis for {original_model_name} (RSVD) - {data_type.title()} Data:")
    print(f"Fisher trace (exact): {stats['fisher_trace']:.6f}")
    if compute_stats:
        print(f"RSVD trace approximation: {stats['rsvd_trace_approximation']:.6f}")
        print(f"Trace approximation ratio: {stats['trace_approximation_ratio']:.4f}" if stats['fisher_trace'] > 0 else "Trace approximation ratio: N/A")
        print(f"Max eigenvalue: {stats['max_eigenvalue']:.6f}")
        print(f"Min eigenvalue: {stats['min_eigenvalue']:.6f}")
        print(f"Condition number: {stats['condition_number']:.6f}")
        print(f"Effective rank: {stats['effective_rank']:.6f}")
    print(f"Number of components (k): {stats['num_components_k']}")
    print(f"Number of parameters: {stats['num_parameters']}")
    if compute_stats:
        print(f"RSVD error bound: {stats['rsvd_error_bound']:.6f}")
    
    # Print timing information
    total_time = time.time() - start_time
    print(f"\nDetailed Timing Summary:")
    print(f"Trace computation: {stats.get('timing_trace_computation', 0.0):.2f} seconds")
    print(f"Step 1 (random matrix): {stats.get('timing_step1_random_matrix', 0.0):.2f} seconds")
    print(f"Step 2 (power iterations): {stats.get('timing_step2_total', 0.0):.2f} seconds")
    print(f"Step 3 (QR decomposition): {stats.get('timing_step3_qr_decomposition', 0.0):.2f} seconds")
    print(f"Step 4 (projection): {stats.get('timing_step4_projection', 0.0):.2f} seconds")
    print(f"Step 5 (SVD): {stats.get('timing_step5_svd', 0.0):.2f} seconds")
    print(f"Step 6 (final eigenvectors): {stats.get('timing_step6_final_eigenvectors', 0.0):.2f} seconds")
    if compute_stats:
        print(f"Analysis and plotting: {stats.get('timing_analysis', 0.0):.2f} seconds")
    print(f"Total RSVD time: {total_rsvd_time:.2f} seconds")
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    import argparse
    from utils.model_loader import (
        load_model_interactive,
        load_model_interactive_untrained,
        load_model_from_trainer,
        load_model_from_trainer_untrained,
        load_best_model,
        load_model_at_epoch,
    )

    parser = argparse.ArgumentParser(description="Fisher Information RSVD Analysis")
    # helper to parse booleans from strings
    def str2bool(v):
        return str(v).lower() in ("1", "true", "t", "yes", "y")
    # Model loading options
    parser.add_argument("--trainer", type=str, default=None, help="Trainer module path, e.g., trainers.specific_trainers.regen_inception")
    parser.add_argument("--checkpoint", type=str, choices=["latest", "best", "epoch"], default="latest", help="Checkpoint selection")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch number if --checkpoint epoch is used")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu; defaults to auto")
    # Analysis options
    parser.add_argument("--data", type=str, choices=["train", "test"], default=None, help="Data split to use")
    parser.add_argument("--k", type=int, default=None, help="Number of RSVD components")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of samples from the chosen split")
    parser.add_argument("--power-iters", type=int, default=1, help="Number of power iterations")
    parser.add_argument("--no-save-intermediates", action="store_true", help="Disable saving intermediate matrices to cache")
    parser.add_argument("--cache-dir", type=str, default=None, help="Root cache directory for intermediates")
    parser.add_argument("--no-stats", action="store_true", help="Skip statistics/plots saving phase")
    parser.add_argument("--use-labels", type=str2bool, default=None, help="Use ground-truth labels (True) or predicted probabilities (False) for loss")
    parser.add_argument("--random-weights", type=str2bool, default=None, help="Initialize model with random weights (do not load checkpoints)")
    parser.add_argument("--weights-seed", type=int, default=None, help="Seed for random model weights (only used when --random-weights is true)")

    args = parser.parse_args()

    # Determine parameters interactively BEFORE loading model
    # Data split
    data_choice = args.data
    if data_choice is None:
        print("\nChoose data for Fisher Information analysis:")
        print("1. Train data")
        print("2. Test data")
        while True:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == '1':
                data_choice = 'train'
                break
            elif choice == '2':
                data_choice = 'test'
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    # Number of components k
    k = args.k
    if k is None:
        while True:
            try:
                k_val = int(input("\nEnter k (number of RSVD components): ").strip())
                if k_val <= 0:
                    print("k must be a positive integer.")
                    continue
                k = k_val
                break
            except ValueError:
                print("Please enter a valid integer.")
    # Use labels?
    use_labels = args.use_labels
    if use_labels is None:
        ul = input("\nUse ground-truth labels for loss? (y/n, default=y): ").strip().lower()
        use_labels = (ul != 'n')
    # Random weights?
    random_weights = args.random_weights
    if random_weights is None:
        rw = input("\nUse random (untrained) model weights? (y/n, default=n): ").strip().lower()
        random_weights = (rw == 'y')
    # Optional weights seed (when random weights are used)
    weights_seed = args.weights_seed
    if random_weights and weights_seed is None:
        ws = input("\nEnter weights seed (integer) or press Enter to skip: ").strip()
        if ws != "":
            try:
                weights_seed = int(ws)
            except ValueError:
                print("Invalid seed; continuing without setting a weights seed.")
                weights_seed = None
    # Optional sample limit
    num_samples = args.num_samples
    if num_samples is None:
        ns = input("\nLimit number of samples? Enter integer or press Enter for all: ").strip()
        if ns != "":
            try:
                num_samples = int(ns)
                if num_samples <= 0:
                    print("Non-positive sample count provided; using all samples.")
                    num_samples = None
            except ValueError:
                print("Invalid input; using all samples.")
                num_samples = None

    # Load model AFTER parameters are set
    # If random weights are requested and a weights seed is provided, set seeds BEFORE importing/instantiating the model
    if random_weights and ('weights_seed' in locals()) and (weights_seed is not None):
        try:
            import torch as _torch_seed_helper
            _torch_seed_helper.manual_seed(weights_seed)
            if _torch_seed_helper.cuda.is_available():
                _torch_seed_helper.cuda.manual_seed_all(weights_seed)
            print(f"Using weights seed: {weights_seed}")
        except Exception as _e:
            print(f"Warning: failed to set weights seed ({weights_seed}): {_e}")
    if args.trainer:
        if random_weights:
            model, model_name, data_loader = load_model_from_trainer_untrained(args.trainer, device=args.device)
        else:
            if args.checkpoint == "latest":
                model, model_name, data_loader = load_model_from_trainer(args.trainer, "model_latest.pt", device=args.device)
            elif args.checkpoint == "best":
                model, model_name, data_loader = load_best_model(args.trainer, device=args.device)
            else:
                if args.epoch is None:
                    raise ValueError("--epoch is required when --checkpoint epoch is used")
                model, model_name, data_loader = load_model_at_epoch(args.trainer, args.epoch, device=args.device)
    else:
        # Interactive fallback
        if random_weights:
            model, model_name, data_loader = load_model_interactive_untrained()
        else:
            model, model_name, data_loader = load_model_interactive()

    # Debug: print first and last 10 weights when using random initialization
    try:
        if random_weights:
            flat_params_dbg = torch.cat([p.detach().view(-1) for p in model.parameters()])
            flat_np = flat_params_dbg[:].cpu().numpy()
            head_vals = flat_np[:10]
            tail_vals = flat_np[-10:] if flat_np.size >= 10 else flat_np
            np.set_printoptions(precision=6, suppress=True)
            print("\n[Weights Debug] First 10 weights:", head_vals)
            print("[Weights Debug] Last 10 weights:", tail_vals)
    except Exception as _dbg_e:
        print(f"[Weights Debug] Failed to print weights: {_dbg_e}")

    # Run analysis
    main(
        model,
        model_name,
        data_loader,
        num_samples=num_samples,
        data_choice=data_choice,
        k=k,
        power_iterations=args.power_iters,
        save_intermediates=(not args.no_save_intermediates),
        compute_stats=(not args.no_stats),
        cache_dir_root=args.cache_dir,
        use_labels=use_labels,
        random_weights=random_weights,
        weights_seed=weights_seed,
    )