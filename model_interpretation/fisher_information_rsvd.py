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


def apply_fisher_to_matrix(model, data_loader, Q, device, side='right', compute_trace=False):
    """
    Implicitly apply the Fisher Information Matrix to matrix Q.
    
    Args:
        model: The PyTorch model
        data_loader: Data loader for the dataset
        Q: Matrix to apply Fisher to (num_params x k)
        device: Device to use for computation
        side: 'right' or 'left' (left will use Q.T)
        compute_trace: If True, also compute the trace of Fisher Information Matrix
        
    Returns:
        Result of Fisher Information Matrix applied to X, error, and optionally trace
    """
    num_params = sum(p.numel() for p in model.parameters())
    k = Q.size(1)
    
    # Initialize result matrix Y = Q.T @ A
    if side == 'right':
        fisher_info_projection = torch.zeros((num_params, k), device=device)
    if side == 'left':
        fisher_info_projection = torch.zeros((k, num_params), device=device)
    error = 0
    fisher_trace = 0.0 if compute_trace else None
    # Keep track of samples processed
    total_samples = 0
    nans_counter = 0
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
                    if torch.isnan(torch.tensor(prob)) or torch.isnan(grad).any():
                        nans_counter += 1
                        continue
                        
                
                        a = 1
                    # Compute trace contribution (sum of squared gradients)
                    if compute_trace:
                        fisher_trace += prob * (grad.T @ grad).item()
                        a = 1
                    
                    if side == 'right':
                        proj = grad.T @ Q # grad @ Q => (1, n) @ (n, k) => (1, k) where n is params, k is projection dim
                        fisher_info_projection.add_(prob * grad @ proj) # (n, 1) @ (1, k) => (n, k)
                        error += prob * (grad.norm(2) - proj.norm(2))
                    if side =='left':
                        proj = Q.T @ grad # Q.T @ grad => (k, n) @ (n, 1) => (k, 1) where n is params, k is projection dim
                        fisher_info_projection.add_(prob * proj @ grad.T) # (k, 1) @ (1, n) => (k, n)
                        error += prob * (grad.norm(2) - proj.norm(2))

                    

    # Print number of samples with NaNs and number of classes (output shape)
    num_outputs = list(model.modules())[-1].out_features
    print(f"Number of samples with nans: {nans_counter} / {total_samples * num_outputs}")
    # Average over total samples
    fisher_info_projection /= total_samples
    error = error / total_samples
    if compute_trace:
        fisher_trace /= total_samples
        return fisher_info_projection, error, fisher_trace
    else:
        return fisher_info_projection, error




def calculate_fisher_rsvd(model, data_loader, k, power_iterations=1, save_intermediates=True, cache_dir="rsvd_cache"):
    """
    Calculate the Fisher Information Matrix using Randomized SVD (RSI algorithm).
    
    Args:
        model: The PyTorch model
        data_loader: Data loader for the dataset
        k: Number of random projections/dimensions to use
        power_iterations: Number of power iterations to enhance accuracy (default: 1)
        save_intermediates: Whether to save intermediate results
        cache_dir: Directory to save intermediate results
        
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
            AX, e = apply_fisher_to_matrix(model, data_loader, X, device, side='right')
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
        B, error, fisher_trace = apply_fisher_to_matrix(model, data_loader, Q, device, side='left', compute_trace=True)
        if save_intermediates:
            np.save(B_file, B.cpu().numpy())
            np.save(error_file, np.array(error.cpu().numpy()))
            np.save(fisher_trace_file, np.array(fisher_trace))
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


def main(model, model_name, data_loader):
    start_time = time.time()
    
    # Let user choose between train and test data
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
    
    print(f"Using {data_type} data for Fisher Information analysis.")
    
    # Let user choose k (number of components)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Choose the number of components (k) for RSVD:")
    print("Recommended values:")
    print("- Small models (<100k params): 100-500")
    print("- Medium models (100k-1M params): 500-1000")
    print("- Large models (>1M params): 1000-2000")
    
    while True:
        try:
            k = int(input("Enter k (number of components): ").strip())
            if k <= 0:
                print("k must be a positive integer.")
                continue
            num_params = sum(p.numel() for p in model.parameters())
            if k >= num_params:
                print(f"k ({k}) must be less than the number of parameters ({num_params}).")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")
    
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
    output_dir = Path(f'model_interpretation/outputs/fisher_analysis/{original_model_name}/')
    
    print(f"\nAnalyzing {original_model_name} (model type: {model_class_name})...")
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Calculate Fisher Information Matrix using RSVD
    fim_start_time = time.time()
    power_iterations = 1  # Number of power iterations (fixed)
    
    # Create cache directory specific to this model and data type
    cache_dir = f"model_interpretation/outputs/rsvd_cache/{original_model_name}_{data_type}_k{k}_p{power_iterations}"
    
    U, S, V, error, fisher_trace, timings = calculate_fisher_rsvd(model, data_for_analysis, k, power_iterations, 
                                                                save_intermediates=True, cache_dir=cache_dir)
    fim_end_time = time.time()
    print("Fisher Information Matrix analysis with RSVD completed.")
    total_rsvd_time = fim_end_time - fim_start_time
    print(f"RSVD calculation took {total_rsvd_time:.2f} seconds")
    
    # Add total RSVD time to timings
    timings['total_rsvd_time'] = total_rsvd_time
    
    # Analyze and save results
    analysis_start_time = time.time()
    stats = analyze_fisher_information_rsvd(U, S, V, model, original_model_name, output_dir, error, fisher_trace, timings, data_type)
    analysis_end_time = time.time()
    
    # Add analysis time to stats
    analysis_time = analysis_end_time - analysis_start_time
    stats['timing_analysis'] = analysis_time
    
    # Re-save statistics with updated timing information
    with open(output_dir / f'{data_type}_{original_model_name}_fisher_stats_rsvd.txt', 'w') as f:
        for key, value in stats.items():
            f.write(f'{key}: {value:.6f}\n')
    
    # Print summary statistics
    print(f"\nFisher Information Analysis for {original_model_name} (RSVD) - {data_type.title()} Data:")
    print(f"Fisher trace (exact): {stats['fisher_trace']:.6f}")
    print(f"RSVD trace approximation: {stats['rsvd_trace_approximation']:.6f}")
    print(f"Trace approximation ratio: {stats['trace_approximation_ratio']:.4f}" if stats['fisher_trace'] > 0 else "Trace approximation ratio: N/A")
    print(f"Max eigenvalue: {stats['max_eigenvalue']:.6f}")
    print(f"Min eigenvalue: {stats['min_eigenvalue']:.6f}")
    print(f"Condition number: {stats['condition_number']:.6f}")
    print(f"Effective rank: {stats['effective_rank']:.6f}")
    print(f"Number of components (k): {stats['num_components_k']}")
    print(f"Number of parameters: {stats['num_parameters']}")
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
    print(f"Analysis and plotting: {stats.get('timing_analysis', 0.0):.2f} seconds")
    print(f"Total RSVD time: {total_rsvd_time:.2f} seconds")
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