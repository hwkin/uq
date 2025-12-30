from torch.nn.utils import parameters_to_vector, vector_to_parameters
from collections import OrderedDict
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
src_path = "/../../"
sys.path.append(src_path + 'plotting/')
from field_plot import field_plot # pyright: ignore[reportMissingImports]

# --- Utilities to flatten/unflatten model parameters ---
def get_param_shapes(model):
    return [p.shape for p in model.parameters()]

def pack_params(model):
    return parameters_to_vector([p.detach() for p in model.parameters()])

def unpack_params(model, flat):
    vec = flat.to(next(model.parameters()).device)
    vector_to_parameters(vec, model.parameters())

# --- Log-posterior: Gaussian prior + Gaussian likelihood for PCANet ---
def make_log_prob_fn(model, X, y, noise_std=0.01, prior_std=1.0):
    """
    Create log probability function for HMC sampling.
    
    Args:
        model: PCANet model
        X: Input data (reduced dimension input), numpy array or tensor
        y: Target output (reduced dimension output), numpy array or tensor
        noise_std: Observation noise standard deviation
        prior_std: Prior standard deviation for weights
    
    Note: PCANet uses X -> Y mapping directly (no trunk network like DeepONet)
    """
    device = next(model.parameters()).device
    named_params = [(name, p) for name, p in model.named_parameters()]
    
    # Convert to tensors if needed
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    
    X = X.to(device)
    y = y.to(device)

    def _flat_to_param_dict(flat_params: torch.Tensor) -> OrderedDict:
        flat_params = flat_params.to(device)
        params = OrderedDict()
        offset = 0
        for name, p in named_params:
            numel = p.numel()
            params[name] = flat_params[offset:offset + numel].view_as(p)
            offset += numel
        if offset != flat_params.numel():
            raise ValueError(
                f"Flat parameter size mismatch: used {offset} elements, got {flat_params.numel()}"
            )
        return params

    def log_prob(flat_params):
        # Use a functional call so the computation graph depends on flat_params.
        # NOTE: model.predict() uses torch.no_grad(); do NOT use it here.
        params = _flat_to_param_dict(flat_params)
        pred = torch.func.functional_call(model, params, (X,))
        resid = (y - pred).reshape(y.shape[0], -1)
        # Log-likelihood (Gaussian)
        ll = -0.5 * (resid.pow(2).sum() / (noise_std**2))
        # Log-prior (Gaussian)
        lp = -0.5 * (flat_params.pow(2).sum() / (prior_std**2))
        return ll + lp
    
    return log_prob

# --- Adaptive HMC with Dual Averaging ---
def hmc_adaptive(log_prob_fn, initial, target_accept=0.75, initial_step_size=1e-6, 
                 leapfrog_steps=10, num_samples=500, burn_in=100, adapt_steps=None, min_step_size=1e-10, max_step_size=1e-5):
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
            step_size = float(np.clip(np.exp(log_eps), min_step_size, max_step_size))
            log_eps = np.log(step_size)
            
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
            samples.append(current.detach().clone())
        
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

def compute_diagonal_hessian(model, X, y, noise_std, prior_std, device, batch_size=20, sample_outputs_per_batch=50):
    """
    Compute diagonal approximation of the Hessian using the Gauss-Newton method.
    This is memory efficient: it processes one sample at a time and only samples
    a subset of output dimensions.
    
    H_diag ≈ sum_i (∂f/∂θ)² / σ² + 1/σ_prior²
    
    NOTE: We use model.forward() instead of model.predict() because predict()
    wraps the forward pass in torch.no_grad(), which disables gradient computation.
    
    For PCANet: X -> Y mapping (no trunk network)
    """
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    
    # Convert inputs to tensors once
    X_tensor = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X.clone()
    y_tensor = torch.from_numpy(y).float() if isinstance(y, np.ndarray) else y.clone()
    
    n_samples = X_tensor.shape[0]
    n_outputs = y_tensor.shape[1] if len(y_tensor.shape) > 1 else 1
    
    # Initialize diagonal Hessian with prior term
    H_diag = torch.ones(n_params, device=device) / (prior_std ** 2)
    
    # Scale factor to account for subsampling output dimensions
    scale_factor = n_outputs / sample_outputs_per_batch
    noise_var_inv = 1.0 / (noise_std ** 2)
    
    # Process samples in batches
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        X_batch = X_tensor[i:batch_end].to(device)
        
        # Sample output dimensions for this batch (different random sample each batch)
        sample_indices = np.random.choice(n_outputs, min(sample_outputs_per_batch, n_outputs), replace=False)
        
        # Forward pass - use forward() instead of predict() to enable gradient computation
        # predict() uses torch.no_grad() which disables gradients
        pred = model.forward(X_batch)  # [batch_size, n_outputs]
        
        # Process each sample and sampled output dimension
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
        del pred, X_batch
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

def uqevaluation(num_test, test_data, model, method, hmc_samples=None, la_samples=None):
    # Define noise standard deviation (should match the value used in HMC sampling)
    noise_std = 0.05
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_indices = np.random.choice(len(test_data["X_train"]), num_test, replace=False)
    eval_indices.sort()
    epoch_mcd = 100

    x_eval = test_data['X_train'][eval_indices]
    y_eval = test_data['Y_train'][eval_indices]

    print(f"Evaluating uncertainty on {num_test} test samples...")

    if method == 'hmc':
        # Compute predictions for each posterior sample
        print("Computing posterior predictions...")
        with torch.no_grad():
            preds_eval_list = []
            for idx, s in enumerate(hmc_samples): # type: ignore
                unpack_params(model, s.to(device))
                x_tensor = torch.from_numpy(x_eval).float().to(device) if isinstance(x_eval, np.ndarray) else x_eval.to(device)
                pred = model.predict(x_tensor)
                preds_eval_list.append(pred.cpu().numpy())
        preds_eval = np.stack(preds_eval_list)  # [n_posterior, n_eval, n_outputs]
    elif method == 'mcd':
        preds_eval_list = []
        with torch.no_grad():
            for i in range(epoch_mcd):
                x_tensor = torch.from_numpy(x_eval).float().to(device)
                # Use forward() instead of predict() to keep dropout active
                pred = model.forward(x_tensor)
                preds_eval_list.append(pred.cpu().numpy())
        preds_eval = np.stack(preds_eval_list)  # [n_eval, n_outputs]
    elif method == 'la':
        preds_eval_list = []
        with torch.no_grad():
            for idx, s in enumerate(la_samples): # type: ignore
                unpack_params(model, s.to(device))
                x_tensor = torch.from_numpy(x_eval).float().to(device) if isinstance(x_eval, np.ndarray) else x_eval.to(device)
                pred = model.predict(x_tensor)
                preds_eval_list.append(pred.cpu().numpy())
        preds_eval = np.stack(preds_eval_list)  # [num_samples, num_eval, num_outputs]

    # Compute uncertainties
    mean_pred_eval = preds_eval.mean(axis=0)
    epistemic_var_eval = preds_eval.var(axis=0)
    epistemic_std_eval = np.sqrt(epistemic_var_eval)
    aleatoric_var_eval = noise_std ** 2
    total_var_eval = epistemic_var_eval + aleatoric_var_eval
    total_std_eval = np.sqrt(total_var_eval)
    sample_std = np.mean(epistemic_std_eval, axis=1)

    # PREDICTION ERROR
    errors = y_eval - mean_pred_eval
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    rmse = np.sqrt(np.mean(squared_errors))

    # CALIBRATION - Check if uncertainties are well-calibrated
    z_scores = np.abs(errors) / total_std_eval

    coverage_1sigma = np.mean(z_scores <= 1.0)  # Should be ~68.3%
    coverage_2sigma = np.mean(z_scores <= 2.0)  # Should be ~95.4%
    coverage_3sigma = np.mean(z_scores <= 3.0)  # Should be ~99.7%

    # SHARPNESS - How tight are the uncertainty bounds?
    num_sigma = 2.0
    widths = 2 * num_sigma * total_std_eval
    mpiw = np.mean(widths)

    # NEGATIVE LOG-LIKELIHOOD (proper scoring rule)
    nll = 0.5 * np.mean(np.log(2 * np.pi * total_var_eval) + squared_errors / total_var_eval)

    return sample_std, np.array([rmse, coverage_1sigma, coverage_2sigma, coverage_3sigma, mpiw, nll, np.mean(total_std_eval)])

def comparison_uq(result1,result2,result3):
    print("="*70)
    print("COMPARISON: HMC vs MC Dropout vs Laplace Approximation")
    print("="*70)

    comparison_data = {
        'Metric': [
            'RMSE', 
            'Coverage 1σ (%)',
            'Coverage 2σ (%)',
            'Coverage 3σ (%)',
            'MPIW',
            'NLL'
        ],
        'HMC': [
            f'{result1[0]:.6f}',
            f'{result1[1]*100:.1f}',
            f'{result1[2]*100:.1f}',
            f'{result1[3]*100:.1f}',
            f'{result1[4]:.4f}',
            f'{result1[5]:.4f}'
        ],
        'MC Dropout': [
            f'{result2[0]:.6f}',
            f'{result2[1]*100:.1f}',
            f'{result2[2]*100:.1f}',
            f'{result2[3]*100:.1f}',
            f'{result2[4]:.4f}',
            f'{result2[5]:.4f}'
        ],
        'Laplace': [
            f'{result3[0]:.6f}',
            f'{result3[1]*100:.1f}',
            f'{result3[2]*100:.1f}',
            f'{result3[3]*100:.1f}',
            f'{result3[4]:.4f}',
            f'{result3[5]:.4f}'
        ],
        'Ideal': [
            'Lower',
            '68.3',
            '95.4',
            '99.7',
            'Lower',
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

def plot_uq(num_test, test_data, model, data, method, hmcsamples=None, lasamples=None):
    # Define visualization parameters
    num_vis_samples = 5  # Number of samples to visualize
    vis_indices = np.random.choice(num_test, num_vis_samples, replace=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    noise_std=0.05

    # Get visualization data
    x_vis = test_data['X_train'][vis_indices]
    y_vis = test_data['Y_train'][vis_indices]
    
    if method == 'hmc':
    # Thin samples for visualization predictions

        # Compute predictions for each posterior sample (for visualization subset)
        print(f"Computing HMC predictions for {num_vis_samples} visualization samples...")
        with torch.no_grad():
            preds_list = []
            for idx, s in enumerate(hmcsamples): # pyright: ignore[reportArgumentType]
                unpack_params(model, s.to(device))
                x_tensor = torch.from_numpy(x_vis).float().to(device) if isinstance(x_vis, np.ndarray) else x_vis.to(device)
                pred = model.predict(x_tensor)
                preds_list.append(data.decoder_Y(pred.cpu().numpy()))
            preds = np.stack(preds_list)  # [n_posterior, n_vis, n_outputs]
    elif method == 'mcd':
        preds_list = []
        with torch.no_grad():
            for i in range(num_vis_samples):
                x_tensor = torch.from_numpy(x_vis).float().to(device) if isinstance(x_vis, np.ndarray) else x_vis.to(device)
                # Use forward() instead of predict() to keep dropout active
                pred = model.forward(x_tensor)
                preds_list.append(data.decoder_Y(pred.cpu().numpy()))
        preds = np.stack(preds_list)
    elif method == 'la':
        preds_list = []
        with torch.no_grad():
            for idx, s in enumerate(lasamples):  # pyright: ignore[reportArgumentType]
                unpack_params(model, s.to(device))
                x_tensor = torch.from_numpy(x_vis).float().to(device) if isinstance(x_vis, np.ndarray) else x_vis.to(device)
                pred = model.predict(x_tensor)
                preds_list.append(data.decoder_Y(pred.cpu().numpy()))
        preds = np.stack(preds_list)  # [num_samples, num_vis, n_outputs]
    

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

    x_vis = data.decoder_X(x_vis)
    y_vis = data.decoder_Y(y_vis)
    mean_pred_vis = preds.mean(axis=0)
    epistemic_var_vis = preds.var(axis=0)
    epistemic_std_vis = np.sqrt(epistemic_var_vis)
    aleatoric_std_vis = noise_std * np.ones_like(mean_pred_vis)
    total_std_vis = np.sqrt(epistemic_var_vis + noise_std**2)

    for i in range(rows):
        i_plot = vis_indices[i]
        
        # Get data (PCANet uses reduced dimension, need to decode)
        i_m = x_vis[i]
        i_truth = y_vis[i]
        i_mean = mean_pred_vis[i]
        i_epistemic = epistemic_std_vis[i]
        i_aleatoric = aleatoric_std_vis[i]
        i_total = total_std_vis[i]

        def apply_dirichlet_bc(u, bc_value, bc_node_ids):
            u[bc_node_ids] = bc_value
            return u
        i_mean = apply_dirichlet_bc(i_mean.copy(), 0.0, data.u_mesh_dirichlet_boundary_nodes)
        
        i_error = np.abs(i_truth - i_mean)
        rel_error = np.linalg.norm(i_error) / np.linalg.norm(i_truth)
        
        uvec = [i_m, i_truth, i_mean, i_epistemic, 
                    i_aleatoric, i_total, i_error]
        
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

def run_regression_shift(method, levels, results):
    stats = {m: {'rmse': [], 'nll': [], 'unc': [], 'cov': []} for m in method}

    for met, result in zip(method, results):
        for i, lvl in enumerate(levels):
            stats[met]['rmse'].append(result[i][0])
            stats[met]['cov'].append(result[i][2])
            stats[met]['mpiw'].append(result[i][4])
            stats[met]['nll'].append(result[i][5])
            stats[met]['unc'].append(result[i][6])

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    metrics = ['rmse', 'cov', 'mpiw', 'nll', 'unc']
    titles = ['RMSE (Error) (↓)', '95% Coverage (Target: 0.95)' , 'MPIW (↓)', 'NLL (↓)', 'Uncertainty (Avg Std) (↑)']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for met, data in stats.items():
            ax.plot(levels, data[metric], marker='o', label=met)
            
        # Draw target line for coverage
        if metric == 'cov':
            ax.axhline(0.95, color='black', linestyle='--', label='Ideal')
            
        ax.set_title(titles[i])
        ax.set_xlabel('Shift Intensity')
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()
        
    plt.tight_layout()
    plt.show()