from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
src_path = "/../../"
sys.path.append(src_path + 'plotting/')
from field_plot import field_plot

# --- Utilities to flatten/unflatten model parameters ---
def get_param_shapes(model):
    return [p.shape for p in model.parameters()]

def pack_params(model):
    return parameters_to_vector([p.detach() for p in model.parameters()])

def unpack_params(model, flat):
    vec = flat.to(next(model.parameters()).device)
    vector_to_parameters(vec, model.parameters())

# --- Log-posterior: Gaussian prior + Gaussian likelihood for DeepONet ---
def make_log_prob_fn(model, x_branch, x_trunk, y, noise_std=0.01, prior_std=1.0):
    """
    Create log probability function for HMC sampling.
    
    Args:
        model: DeepONet model
        x_branch: Branch network input (input functions), numpy array or tensor
        x_trunk: Trunk network input (evaluation coordinates), numpy array or tensor
        y: Target output, numpy array or tensor
        noise_std: Observation noise standard deviation
        prior_std: Prior standard deviation for weights
    """
    device = next(model.parameters()).device
    
    # Convert to tensors if needed
    if isinstance(x_branch, np.ndarray):
        x_branch = torch.from_numpy(x_branch).float()
    if isinstance(x_trunk, np.ndarray):
        x_trunk = torch.from_numpy(x_trunk).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    
    x_branch = x_branch.to(device)
    x_trunk = x_trunk.to(device)
    y = y.to(device)

    def log_prob(flat_params):
        unpack_params(model, flat_params)
        pred = model.predict(x_branch, x_trunk)  # DeepONet predict method
        resid = (y - pred).reshape(y.shape[0], -1)
        # Log-likelihood (Gaussian)
        ll = -0.5 * (resid.pow(2).sum() / (noise_std**2))
        # Log-prior (Gaussian)
        lp = -0.5 * (flat_params.pow(2).sum() / (prior_std**2))
        return ll + lp
    
    return log_prob

# --- Adaptive HMC with Dual Averaging ---
def hmc_adaptive(log_prob_fn, initial, target_accept=0.75, initial_step_size=1e-6, 
                 leapfrog_steps=10, num_samples=500, burn_in=100, adapt_steps=None):
    """
    Hamiltonian Monte Carlo sampler with dual averaging step size adaptation.
    
    Args:
        log_prob_fn: Function that computes log probability given flat parameters
        initial: Initial parameter vector (requires_grad=True)
        target_accept: Target acceptance rate (0.65-0.80 is optimal for HMC)
        initial_step_size: Initial leapfrog step size
        leapfrog_steps: Number of leapfrog steps per iteration
        num_samples: Number of samples to collect after burn-in
        burn_in: Number of burn-in iterations
        adapt_steps: Number of steps to adapt step size (default: 80% of burn_in)
    
    Returns:
        samples: Tensor of shape (num_samples, dim)
        accept_rate: Final acceptance rate
        final_step_size: Adapted step size
    """
    if adapt_steps is None:
        adapt_steps = int(0.8 * burn_in)
    
    samples = []
    current = initial.clone().detach().requires_grad_(True)
    
    # Dual averaging parameters (from NUTS paper, Hoffman & Gelman 2014)
    mu = np.log(10 * initial_step_size)  # Point to shrink towards
    log_eps = np.log(initial_step_size)
    log_eps_bar = 0.0
    H_bar = 0.0
    gamma = 0.05  # Controls shrinkage towards mu
    t0 = 10       # Stabilization parameter
    kappa = 0.75  # Controls decay rate of adaptation
    
    step_size = initial_step_size
    
    # Compute initial log prob
    current_lp = log_prob_fn(current)
    current_lp_val = current_lp.detach()

    def leapfrog(q, p, eps):
        """
        Leapfrog integrator (Störmer-Verlet) with given step size.
        """
        q = q.clone().detach().requires_grad_(True)
        
        for _ in range(leapfrog_steps):
            # Half step for momentum
            lp = log_prob_fn(q)
            grad = torch.autograd.grad(lp, q, create_graph=False)[0]
            p = p + 0.5 * eps * grad
            
            # Full step for position
            q = (q + eps * p).detach().requires_grad_(True)
            
            # Half step for momentum
            lp = log_prob_fn(q)
            grad = torch.autograd.grad(lp, q, create_graph=False)[0]
            p = p + 0.5 * eps * grad
        
        return q, p, lp.detach()

    accept_count = 0
    total_iterations = num_samples + burn_in
    step_size_history = []
    
    print(f"Starting adaptive HMC with target acceptance rate: {target_accept:.2%}")
    print(f"Adaptation will run for {adapt_steps} iterations")
    
    for i in range(total_iterations):
        # Sample momentum from standard normal
        p0 = torch.randn_like(current)
        
        # Leapfrog integration
        q_prop, p_prop, lp_prop = leapfrog(current, p0, step_size)
        
        # Compute Hamiltonians (H = -log_prob + kinetic energy)
        current_H = -current_lp_val + 0.5 * p0.pow(2).sum()
        prop_H = -lp_prop + 0.5 * p_prop.pow(2).sum()
        
        # Compute acceptance probability
        log_accept_prob = current_H - prop_H
        accept_prob = min(1.0, torch.exp(log_accept_prob).item())
        
        # Handle NaN (reject if energy is NaN)
        if np.isnan(accept_prob):
            accept_prob = 0.0
        
        # Metropolis acceptance step
        if torch.rand(()).item() < accept_prob:
            current = q_prop.detach().requires_grad_(True)
            current_lp_val = lp_prop
            accept_count += 1
        
        # Dual averaging step size adaptation during warm-up
        if i < adapt_steps:
            m = i + 1
            w = 1.0 / (m + t0)
            H_bar = (1 - w) * H_bar + w * (target_accept - accept_prob)
            log_eps = mu - np.sqrt(m) / gamma * H_bar
            step_size = np.exp(log_eps)
            
            # Update averaged step size
            m_power = m ** (-kappa)
            log_eps_bar = m_power * log_eps + (1 - m_power) * log_eps_bar
            
        elif i == adapt_steps:
            # Fix step size to averaged value after adaptation
            step_size = np.exp(log_eps_bar)
            print(f"\n>>> Adaptation complete! Final step size: {step_size:.2e}")
            print(f">>> Acceptance rate during adaptation: {accept_count / (i+1):.3f}\n")
        
        step_size_history.append(step_size)
        
        # Collect samples after burn-in
        if i >= burn_in:
            samples.append(current.detach().cpu())
        
        # Progress reporting
        if (i + 1) % 50 == 0:
            current_accept_rate = accept_count / (i + 1)
            phase = "adapting" if i < adapt_steps else ("burn-in" if i < burn_in else "sampling")
            print(f"Iter {i+1:4d}/{total_iterations}: accept rate = {current_accept_rate:.3f}, "
                  f"step_size = {step_size:.2e}, phase = {phase}")
    
    final_accept_rate = accept_count / total_iterations
    
    return torch.stack(samples), final_accept_rate, step_size, step_size_history


# ============================================================
# Compute the Hessian using Gauss-Newton approximation
# For regression with Gaussian likelihood:
#   H ≈ J^T J / σ² + I / σ_prior²
# where J is the Jacobian of the network output w.r.t. parameters
# ============================================================

def compute_diagonal_hessian(model, x_branch, x_trunk, y, noise_std, prior_std, device, batch_size=20, sample_points_per_batch=50):
    """
    Compute diagonal approximation of the Hessian using the Gauss-Newton method.
    This is memory efficient: it processes one sample at a time and only samples
    a subset of output points.
    
    H_diag ≈ sum_i (∂f/∂θ)² / σ² + 1/σ_prior²
    
    NOTE: We use model.forward() instead of model.predict() because predict()
    wraps the forward pass in torch.no_grad(), which disables gradient computation.
    
    For multi-component outputs (e.g., num_Y_components=2 for 2D displacement),
    the output shape is [batch_size, n_points * num_Y_components]. We sample
    from all output dimensions to get proper Hessian estimates.
    """
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    
    # Convert inputs to tensors once
    x_b = torch.from_numpy(x_branch).float() if isinstance(x_branch, np.ndarray) else x_branch.clone()
    x_t = torch.from_numpy(x_trunk).float().to(device) if isinstance(x_trunk, np.ndarray) else x_trunk.to(device)
    y_tensor = torch.from_numpy(y).float() if isinstance(y, np.ndarray) else y.clone()
    
    n_samples = x_b.shape[0]
    n_points = x_t.shape[0]
    
    # Get the actual output dimension (accounts for multi-component outputs like 2D displacement)
    # Do a test forward pass to determine output shape
    with torch.no_grad():
        test_pred = model.forward(x_b[0:1].to(device), x_t)
        n_outputs = test_pred.shape[1]  # Total output dimension (n_points * num_Y_components)
    
    # Initialize diagonal Hessian with prior term
    H_diag = torch.ones(n_params, device=device) / (prior_std ** 2)
    
    # Scale factor to account for subsampling output points
    # Use n_outputs (actual output dimension) instead of n_points
    scale_factor = n_outputs / sample_points_per_batch
    noise_var_inv = 1.0 / (noise_std ** 2)
    
    # Process samples in batches
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        x_b_batch = x_b[i:batch_end].to(device)
        
        # Sample output indices from ALL output dimensions (not just n_points)
        # This ensures we sample from both ux and uy components for 2D displacement
        sample_indices = np.random.choice(n_outputs, min(sample_points_per_batch, n_outputs), replace=False)
        
        # Forward pass - use forward() instead of predict() to enable gradient computation
        # predict() uses torch.no_grad() which disables gradients
        pred = model.forward(x_b_batch, x_t)  # [batch_size, n_outputs] where n_outputs = n_points * num_Y_components
        
        # Process each sample and sampled output point
        batch_size_actual = pred.shape[0]
        for j in range(batch_size_actual):
            for idx, k in enumerate(sample_indices):
                # Zero gradients before backward
                model.zero_grad()
                
                # Compute gradient for this specific output
                # Use retain_graph only when not at the last iteration
                is_last = (j == batch_size_actual - 1) and (idx == len(sample_indices) - 1)
                pred[j, k].backward(retain_graph=not is_last)
                
                # Accumulate squared gradients (diagonal Hessian approximation)
                grad_sq_sum = torch.zeros(n_params, device=device)
                offset = 0
                for p in params:
                    numel = p.numel()
                    if p.grad is not None:
                        grad_sq_sum[offset:offset + numel] = p.grad.view(-1).pow(2)
                    offset += numel
                
                H_diag += grad_sq_sum * noise_var_inv * scale_factor
        
        # Explicitly delete tensors to free memory
        del pred, x_b_batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if (i + batch_size) % 100 == 0 or batch_end == n_samples:
            print(f"  Processed {batch_end}/{n_samples} samples")
    
    return H_diag

def inject_dropout(model, target_layer_type=nn.Linear, dropout_rate=0.1):
    """
    Recursively adds a Dropout layer after every occurrence of `target_layer_type`.
    """
    for name, child in model.named_children():
        # If the child is the target type (e.g., Linear)
        if isinstance(child, target_layer_type):
            # Create a new Sequential container: [Original Layer, Dropout]
            new_layer = nn.Sequential(
                child, 
                nn.Dropout(dropout_rate)
            )
            # Replace the old child with the new wrapper
            setattr(model, name, new_layer)
        # If the child is a container (Sequential, ModuleList, or custom block), recurse
        else:
            inject_dropout(child, target_layer_type, dropout_rate)

def uqevaluation(num_test, test_data, model, method, hmcsamples=None, lasamples=None):

    # Define noise standard deviation (should match the value used in HMC sampling)
    noise_std = 0.05
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use a larger subset for statistical evaluation
    num_eval_samples = min(200, num_test)  # Evaluate on 200 samples
    eval_indices = np.random.choice(num_test, num_eval_samples, replace=False)
    epoch_mcd = 100

    x_branch_eval = test_data['X_train'][eval_indices]
    x_trunk_eval = test_data['X_trunk']
    y_eval = test_data['Y_train'][eval_indices]

    print(f"Evaluating uncertainty on {num_eval_samples} test samples...")
    
    if method == 'hmc':
        # Compute predictions for each posterior sample
        print("Computing posterior predictions...")
        with torch.no_grad():
            preds_eval_list = []
            for idx, s in enumerate(hmcsamples):
                unpack_params(model, s.to(device))
                x_b = torch.from_numpy(x_branch_eval).float().to(device)
                x_t = torch.from_numpy(x_trunk_eval).float().to(device)
                pred = model.predict(x_b, x_t)
                preds_eval_list.append(pred.cpu().numpy())
        preds_eval = np.stack(preds_eval_list)
    elif method == 'mcd':    
        # Compute MC Dropout predictions
        print("Computing MC Dropout predictions...")
        preds_eval_list = []
        with torch.no_grad():
            for i in range(epoch_mcd):
                x_b = torch.from_numpy(x_branch_eval).float().to(device)
                x_t = torch.from_numpy(x_trunk_eval).float().to(device)
                pred = model.forward(x_b, x_t)
                preds_eval_list.append(pred.cpu().numpy())
        preds_eval = np.stack(preds_eval_list)
    elif method == 'la':
        # Compute Laplace predictions for evaluation set
        print("Computing Laplace posterior predictions...")
        preds_eval_list = []
        with torch.no_grad():
            for idx, s in enumerate(lasamples):
                unpack_params(model, s.to(device))
                x_b = torch.from_numpy(x_branch_eval).float().to(device)
                x_t = torch.from_numpy(x_trunk_eval).float().to(device)
                pred = model.predict(x_b, x_t)
                preds_eval_list.append(pred.cpu().numpy())
        preds_eval = np.stack(preds_eval_list)

    # Compute uncertainties
    mean_pred_eval = preds_eval.mean(axis=0)
    epistemic_var_eval = preds_eval.var(axis=0)
    epistemic_std_eval = np.sqrt(epistemic_var_eval)
    aleatoric_var_eval = noise_std ** 2
    total_var_eval = epistemic_var_eval + aleatoric_var_eval
    total_std_eval = np.sqrt(total_var_eval)

    # ============================================================
    # Uncertainty Quality Metrics
    # ============================================================
    print("\n" + "="*60)
    print("Uncertainty Quality Metrics")
    print("="*60)

    # 1. PREDICTION ERROR
    errors = y_eval - mean_pred_eval
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    rmse = np.sqrt(np.mean(squared_errors))
    mae = np.mean(abs_errors)
    rel_l2_errors = np.linalg.norm(errors, axis=1) / np.linalg.norm(y_eval, axis=1)

    print(f"\n1. PREDICTION ACCURACY:")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   Mean Relative L2 Error: {np.mean(rel_l2_errors)*100:.2f}%")
    print(f"   Std Relative L2 Error:  {np.std(rel_l2_errors)*100:.2f}%")

    # 2. CALIBRATION - Check if uncertainties are well-calibrated
    # For a well-calibrated model: |error| / σ_total should follow N(0,1)
    # So ~68% should be within 1σ, ~95% within 2σ, ~99.7% within 3σ
    z_scores = np.abs(errors) / total_std_eval

    coverage_1sigma = np.mean(z_scores <= 1.0)  # Should be ~68.3%
    coverage_2sigma = np.mean(z_scores <= 2.0)  # Should be ~95.4%
    coverage_3sigma = np.mean(z_scores <= 3.0)  # Should be ~99.7%

    print(f"\n2. CALIBRATION (Coverage Analysis):")
    print(f"   Coverage within 1σ: {coverage_1sigma*100:.1f}% (ideal: 68.3%)")
    print(f"   Coverage within 2σ: {coverage_2sigma*100:.1f}% (ideal: 95.4%)")
    print(f"   Coverage within 3σ: {coverage_3sigma*100:.1f}% (ideal: 99.7%)")

    # Calibration quality indicator
    if coverage_2sigma > 0.99:
        calib_status = "OVER-CONFIDENT (uncertainties too small)"
    elif coverage_2sigma < 0.90:
        calib_status = "UNDER-CONFIDENT (uncertainties too large)"
    else:
        calib_status = "WELL-CALIBRATED"
    print(f"   Status: {calib_status}")

    # 3. SHARPNESS - How tight are the uncertainty bounds?
    mean_epistemic = epistemic_std_eval.mean()
    mean_total = total_std_eval.mean()

    print(f"\n3. SHARPNESS (Uncertainty Magnitude):")
    print(f"   Mean Epistemic σ: {mean_epistemic:.6f}")
    print(f"   Mean Total σ:     {mean_total:.6f}")
    print(f"   Mean Aleatoric σ: {np.sqrt(aleatoric_var_eval):.6f} (fixed)")

    # 4. UNCERTAINTY-ERROR CORRELATION
    # Good uncertainty should correlate with actual errors
    per_sample_epistemic = epistemic_std_eval.mean(axis=1)  # Mean uncertainty per sample
    per_sample_error = abs_errors.mean(axis=1)  # Mean error per sample

    correlation = np.corrcoef(per_sample_epistemic, per_sample_error)[0, 1]

    print(f"\n4. UNCERTAINTY-ERROR CORRELATION:")
    print(f"   Pearson correlation: {correlation:.3f}")
    if correlation > 0.5:
        print(f"   → Good! High uncertainty correlates with high error")
    elif correlation > 0.2:
        print(f"   → Moderate correlation")
    else:
        print(f"   → Weak correlation - uncertainty may not be informative")

    # 5. EPISTEMIC vs ALEATORIC DECOMPOSITION
    epistemic_fraction = epistemic_var_eval / (total_var_eval + 1e-10)
    mean_epistemic_fraction = epistemic_fraction.mean()

    print(f"\n5. UNCERTAINTY DECOMPOSITION:")
    print(f"   Epistemic fraction: {mean_epistemic_fraction*100:.1f}%")
    print(f"   Aleatoric fraction: {(1-mean_epistemic_fraction)*100:.1f}%")
    if mean_epistemic_fraction > 0.8:
        print(f"   → Model uncertainty dominates (more data may help)")
    elif mean_epistemic_fraction < 0.2:
        print(f"   → Data noise dominates (model is confident)")
    else:
        print(f"   → Balanced uncertainty sources")

    # 6. NEGATIVE LOG-LIKELIHOOD (proper scoring rule)
    nll = 0.5 * np.mean(np.log(2 * np.pi * total_var_eval) + squared_errors / total_var_eval)
    print(f"\n6. PROPER SCORING RULES:")
    print(f"   Negative Log-Likelihood: {nll:.4f}")
    print(f"   (Lower is better)")

    print("\n" + "="*60)

    return total_std_eval, [rmse, mae, rel_l2_errors, coverage_1sigma, coverage_2sigma, coverage_3sigma, mean_epistemic, mean_total, mean_epistemic_fraction ,correlation, nll]

def plot_uq(num_test, test_data, model, data, method, hmcsamples=None, lasamples=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise_std = 0.05
    # Define visualization parameters
    num_vis_samples = 5  # Number of samples to visualize
    vis_indices = np.random.choice(num_test, num_vis_samples, replace=False)

    # Get visualization data
    x_branch_vis = test_data['X_train'][vis_indices]
    x_trunk_vis = test_data['X_trunk']
    y_vis = test_data['Y_train'][vis_indices]

    if method == 'hmc':
        # Thin samples for visualization predictions
        thin_factor = max(1, len(hmcsamples) // 30)
        thinned_samples = hmcsamples[::thin_factor]

        with torch.no_grad():
            preds_list = []
            for idx, s in enumerate(thinned_samples):
                unpack_params(model, s.to(device))
                x_b = torch.from_numpy(x_branch_vis).float().to(device)
                x_t = torch.from_numpy(x_trunk_vis).float().to(device)
                pred = model.predict(x_b, x_t)
                preds_list.append(pred.cpu().numpy())
        preds = np.stack(preds_list)
    elif method == 'mcd':
        # Ensure dropout is enabled
        inject_dropout(model)
        torch.nn.Module.train(model)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                torch.nn.Module.train(module)        
        # Compute MC Dropout predictions
        preds_list = []
        with torch.no_grad():
            for i in range(num_vis_samples):
                x_b = torch.from_numpy(x_branch_vis).float().to(device)
                x_t = torch.from_numpy(x_trunk_vis).float().to(device)
                pred = model.predict(x_b, x_t)
                preds_list.append(pred.cpu().numpy())
        preds = np.stack(preds_list)
    elif method == 'la':
        # Compute Laplace predictions for visualization set
        preds_list = []
        with torch.no_grad():
            for idx, s in enumerate(lasamples):
                unpack_params(model, s.to(device))
                x_b = torch.from_numpy(x_branch_vis).float().to(device)
                x_t = torch.from_numpy(x_trunk_vis).float().to(device)
                pred = model.predict(x_b, x_t)
                preds_list.append(pred.cpu().numpy())
        preds = np.stack(preds_list)
    else:
        raise ValueError("Invalid method for UQ plotting")
    
    if preds is None:
        raise ValueError("Predictions not computed for UQ plotting")
    mean_pred_vis = preds.mean(axis=0)
    epistemic_var_vis = preds.var(axis=0)
    epistemic_std_vis = np.sqrt(epistemic_var_vis)
    aleatoric_std_vis = noise_std * np.ones_like(mean_pred_vis)
    total_std_vis = np.sqrt(epistemic_var_vis + noise_std**2)

    print(f"Predictions shape: {preds.shape}")

    # Set up plotting
    rows = num_vis_samples
    cols = 7  # m, u_true, u_mean, σ_epistemic, σ_aleatoric, σ_total, error
    fs = 14

    fig, axs = plt.subplots(rows, cols, figsize=(28, 4*rows))

    nodes = data.X_trunk
    u_tags = [r'$m$ (input)', r'$u_{true}$', r'$u_{mean}$', 
                r'$\sigma_{epistemic}$', r'$\sigma_{aleatoric}$', 
                r'$\sigma_{total}$', r'$|u_{true} - u_{mean}|$']
    cmaps = ['jet', 'viridis', 'viridis', 'plasma', 'plasma', 'plasma', 'hot']

    for i in range(rows):
        # Get data
        i_m = x_branch_vis[i]
        i_truth = y_vis[i]
        i_mean = mean_pred_vis[i]
        i_epistemic = epistemic_std_vis[i]
        i_aleatoric = aleatoric_std_vis[i]
        i_total = total_std_vis[i]

        i_m = data.decoder_X(i_m)
        i_truth = data.decoder_Y(i_truth)
        i_mean_decoded = data.decoder_Y(i_mean)
        scale_factor = data.std_Y if hasattr(data, 'std_Y') else 1.0
        i_epistemic_scaled = i_epistemic * scale_factor
        i_aleatoric_scaled = i_aleatoric * scale_factor
        i_total_scaled = i_total * scale_factor
        
        # Apply Dirichlet BC
        def apply_dirichlet_bc(u, bc_value, bc_node_ids):
            u[bc_node_ids] = bc_value
            return u
        i_mean_decoded = apply_dirichlet_bc(i_mean_decoded.copy(), 0.0, data.u_mesh_dirichlet_boundary_nodes)
        
        i_error = np.abs(i_truth - i_mean_decoded)
        rel_error = np.linalg.norm(i_error) / np.linalg.norm(i_truth)
        
        uvec = [i_m, i_truth, i_mean_decoded, i_epistemic_scaled, 
                    i_aleatoric_scaled, i_total_scaled, i_error]
        
        for j in range(cols):
            cbar = field_plot(axs[i,j], uvec[j], nodes, cmap=cmaps[j])
            
            divider = make_axes_locatable(axs[i,j])
            cax = divider.append_axes('right', size='8%', pad=0.03)
            cax.tick_params(labelsize=fs-4)
            
            kfmt = lambda x, pos: "{:.2g}".format(x)
            fig.colorbar(cbar, cax=cax, orientation='vertical', format=kfmt)
            
            if i == 0:
                axs[i,j].set_title(u_tags[j], fontsize=fs)
            
            if j == cols - 1:
                axs[i,j].set_title(f'rel err: {rel_error*100:.2f}%', fontsize=fs-2)
            
            axs[i,j].axis('off')

    fig.tight_layout()
    plt.show()

def comparison_uq(result1,result2,result3):
    print("="*70)
    print("COMPARISON: HMC vs MC Dropout vs Laplace Approximation")
    print("="*70)

    comparison_data = {
        'Metric': [
            'RMSE', 
            'MAE', 
            'Mean Rel. L2 Error (%)',
            'Coverage 1σ (%)',
            'Coverage 2σ (%)',
            'Coverage 3σ (%)',
            'Mean Epistemic σ',
            'Mean Total σ',
            'Epistemic Fraction (%)',
            'Uncertainty-Error Corr.',
            'NLL'
        ],
        'HMC': [
            f'{result1[0]:.6f}',
            f'{result1[1]:.6f}',
            f'{np.mean(result1[2])*100:.2f}',
            f'{result1[3]*100:.1f}',
            f'{result1[4]*100:.1f}',
            f'{result1[5]*100:.1f}',
            f'{result1[6]:.6f}',
            f'{result1[7]:.6f}',
            f'{result1[8]*100:.1f}',
            f'{result1[9]:.3f}',
            f'{result1[10]:.4f}'
        ],
        'MC Dropout': [
            f'{result2[0]:.6f}',
            f'{result2[1]:.6f}',
            f'{np.mean(result2[2])*100:.2f}',
            f'{result2[3]*100:.1f}',
            f'{result2[4]*100:.1f}',
            f'{result2[5]*100:.1f}',
            f'{result2[6]:.6f}',
            f'{result2[7]:.6f}',
            f'{result2[8]*100:.1f}',
            f'{result2[9]:.3f}',
            f'{result2[10]:.4f}'
        ],
        'Laplace': [
            f'{result3[0]:.6f}',
            f'{result3[1]:.6f}',
            f'{np.mean(result3[2])*100:.2f}',
            f'{result3[3]*100:.1f}',
            f'{result3[4]*100:.1f}',
            f'{result3[5]*100:.1f}',
            f'{result3[6]:.6f}',
            f'{result3[7]:.6f}',
            f'{result3[8]*100:.1f}',
            f'{result3[9]:.3f}',
            f'{result3[10]:.4f}'
        ],
        'Ideal': [
            'Lower',
            'Lower',
            'Lower',
            '68.3',
            '95.4',
            '99.7',
            '-',
            '-',
            '-',
            'Higher',
            'Lower'
        ]
    }

    # Print comparison table
    print("\n{:<25} {:>12} {:>12} {:>12} {:>10}".format('Metric', 'HMC', 'MC Dropout', 'Laplace', 'Ideal'))
    print("-" * 85)
    for i in range(len(comparison_data['Metric'])):
        print("{:<25} {:>12} {:>12} {:>12} {:>10}".format(
            comparison_data['Metric'][i],
            comparison_data['HMC'][i],
            comparison_data['MC Dropout'][i],
            comparison_data['Laplace'][i],
            comparison_data['Ideal'][i]
        ))

    # Create a bar chart comparison
    fig_comp, axes_comp = plt.subplots(2, 3, figsize=(15, 8))

    methods = ['HMC', 'MC Dropout', 'Laplace']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    # Plot 1: Calibration (2σ coverage)
    ax = axes_comp[0, 0]
    coverages = [result1[4]*100, result2[4]*100, result3[4]*100]
    bars = ax.bar(methods, coverages, color=colors)
    ax.axhline(y=95.4, color='black', linestyle='--', linewidth=2, label='Ideal (95.4%)')
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('2σ Coverage (Calibration)', fontsize=14)
    ax.set_ylim([0, 105])
    ax.legend()

    # Plot 2: Mean Epistemic σ
    ax = axes_comp[0, 1]
    epistemic_sigmas = [result1[6], result2[6], result3[6]]
    ax.bar(methods, epistemic_sigmas, color=colors)
    ax.set_ylabel('Mean σ_epistemic', fontsize=12)
    ax.set_title('Epistemic Uncertainty', fontsize=14)

    # Plot 3: Uncertainty-Error Correlation
    ax = axes_comp[0, 2]
    correlations = [result1[9], result2[9], result3[9]]
    ax.bar(methods, correlations, color=colors)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Uncertainty-Error Correlation', fontsize=14)
    ax.set_ylim([0, 1])

    # Plot 4: NLL
    ax = axes_comp[1, 0]
    nlls = [result1[10], result2[10], result3[10]]
    ax.bar(methods, nlls, color=colors)
    ax.set_ylabel('NLL', fontsize=12)
    ax.set_title('Negative Log-Likelihood (Lower is Better)', fontsize=14)

    # Plot 5: Epistemic Fraction
    ax = axes_comp[1, 1]
    epi_fracs = [result1[8]*100, result2[8]*100, result3[8]*100]
    ax.bar(methods, epi_fracs, color=colors)
    ax.set_ylabel('Epistemic Fraction (%)', fontsize=12)
    ax.set_title('Epistemic vs Total Uncertainty', fontsize=14)

    # Plot 6: Mean Relative L2 Error
    ax = axes_comp[1, 2]
    rel_errors = [np.mean(result1[2])*100, np.mean(result2[2])*100, np.mean(result3[2])*100]
    ax.bar(methods, rel_errors, color=colors)
    ax.set_ylabel('Mean Rel. L2 Error (%)', fontsize=12)
    ax.set_title('Prediction Accuracy', fontsize=14)

    fig_comp.tight_layout()
    fig_comp.suptitle('Comparison of Uncertainty Quantification Methods', fontsize=16, y=1.02)
    plt.show()
