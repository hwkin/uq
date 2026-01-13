from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch
import torch.nn as nn
import numpy as np
import hamiltorch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.append(".")
import uq_evaluation

# --- Utilities to flatten/unflatten model parameters ---
def get_param_shapes(model):
    return [p.shape for p in model.parameters() if p.requires_grad]

def pack_params(model):
    return parameters_to_vector([p.detach() for p in model.parameters() if p.requires_grad])

def unpack_params(model, flat):
    vec = flat.to(next(model.parameters()).device)
    vector_to_parameters(vec, [p for p in model.parameters() if p.requires_grad])

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
        pred = model.forward(x_branch, x_trunk)  # Use forward to keep gradients
        resid = (y - pred).reshape(y.shape[0], -1)
        # Log-likelihood (Gaussian)
        ll = -0.5 * (resid.pow(2).sum() / (noise_std**2))
        # Log-prior (Gaussian)
        lp = -0.5 * (flat_params.pow(2).sum() / (prior_std**2))
        return ll + lp
    
    return log_prob

def make_minibatch_log_prob_fn(model, x_branch, x_trunk, y, batch_size=100, noise_std=0.01, prior_std=1.0):
    """
    Create stochastic log probability function for SGLD with mini-batching.
    To allow stochastic gradient estimation, this function samples a mini-batch
    inside the returned log_prob function and scales the likelihood accordingly.
    
    Args:
        model: DeepONet model
        x_branch: Branch network input (all training samples)
        x_trunk: Trunk network input
        y: Target output (all training samples)
        batch_size: Mini-batch size
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
    
    # Don't move full dataset to GPU yet if it's too large, but here we assume it fits.
    x_branch = x_branch.to(device)
    x_trunk = x_trunk.to(device)
    y = y.to(device)
    
    N = x_branch.shape[0]

    def log_prob(flat_params):
        unpack_params(model, flat_params)
        
        # Sample mini-batch
        idx = torch.randperm(N, device=device)[:batch_size]
        x_b_batch = x_branch[idx]
        y_batch = y[idx]
        
        pred = model.forward(x_b_batch, x_trunk) # Use forward to keep gradients
        resid = (y_batch - pred).reshape(batch_size, -1)
        
        # Log-likelihood (Gaussian) - SCALED by N/batch_size
        sse = resid.pow(2).sum()
        ll = -0.5 * (N / batch_size) * (sse / (noise_std**2))
        
        # Log-prior (Gaussian) - NOT scaled (applied once to parameters)
        lp = -0.5 * (flat_params.pow(2).sum() / (prior_std**2))
        
        return ll + lp
    
    return log_prob

def freezelayer(model, device):
    # Freeze all parameters except the last layer ---
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last layer of branch net
    for param in model.branch_net.layers[-1].parameters():
        param.requires_grad = True
    # Unfreeze last layer of trunk net
    for param in model.trunk_net.layers[-1].parameters():
        param.requires_grad = True
    # Unfreeze bias
    for param in model.bias.parameters():
        param.requires_grad = True
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # Initialize from current model parameters (trainable subset only)
    flat0 = pack_params(model).to(device)
    print(f"Initial parameter vector shape (trainable only): {flat0.shape}")
    # Std of the parameters that will be sampled (current last-layer weights)
    param_std_last_init = torch.std(flat0)
    print(f"Initial last-layer param std (scalar): {param_std_last_init.item():.3e}")
    return model, flat0, param_std_last_init.item()

def build_full_vector(model, base_state, all_params, trainable_flat):
    offset = 0
    with torch.no_grad():
        # restore frozen params to base
        for name, param in model.named_parameters():
            param.copy_(base_state[name])
        # insert sampled trainable params
        for p in all_params:
            if p.requires_grad:
                numel = p.numel()
                p.copy_(trainable_flat[offset:offset + numel].view_as(p))
                offset += numel
    return parameters_to_vector(all_params).detach().cpu()

# --- HMC using Hamiltorch ---
def hmc_nuts(log_prob_fn, initial, initial_step_size=1e-4, leapfrog_steps=20, num_samples=1000, burn_in=1000, random_seed=42):
    """
    Hamiltonian Monte Carlo sampler using Hamiltorch.
    """
    print(f"Starting HMC with hamiltorch...")
    print(f"  Samples: {num_samples} + Burn-in: {burn_in}")
    hamiltorch.set_random_seed(random_seed)
    
    # hamiltorch.sample() returns a list of sample tensors
    params_hmc = hamiltorch.sample(log_prob_func=log_prob_fn, 
                                   params_init=initial, 
                                   num_samples=num_samples, 
                                   burn=burn_in,
                                   step_size=initial_step_size, 
                                   num_steps_per_sample=leapfrog_steps,
                                   sampler=hamiltorch.Sampler.HMC_NUTS)
    
    samples = torch.stack(params_hmc)
    return samples

def sgld(log_prob_fn, initial, step_size=5e-5, num_samples=2000, burn_in=2000,
         step_decay=0.9999, min_step_size=1e-7, grad_clip=100.0, random_seed=42):
    """
    Stochastic Gradient Langevin Dynamics (SGLD) sampler.
    
    SGLD is better suited for mini-batch settings than standard HMC because it
    doesn't require Hamiltonian conservation. It adds noise to SGD updates.
    
    Args:
        log_prob_fn: Function that computes log probability given flat parameters
        initial: Initial parameter vector (requires_grad=True)
        step_size: Initial step size (learning rate)
        num_samples: Number of samples to collect after burn-in
        burn_in: Number of burn-in iterations
        step_decay: Multiplicative decay factor for step size per iteration
        min_step_size: Minimum step size (to prevent collapse)
        grad_clip: Maximum gradient norm for clipping (to prevent explosion)
        random_seed: Random seed for reproducibility
    
    Returns:
        samples: List of parameter samples
        final_step_size: Final step size
    """
    torch.manual_seed(random_seed)
    samples = []
    current = initial.clone().detach().requires_grad_(True)
    
    total_iterations = num_samples + burn_in
    eps = step_size
    
    print(f"Starting SGLD sampling...")
    print(f"  Burn-in: {burn_in}, Samples: {num_samples}")
    print(f"  Initial step size: {step_size:.2e}")
    
    for i in range(total_iterations):
        # Compute gradient of log probability
        lp = log_prob_fn(current)
        if not torch.isfinite(lp):
            print(f"Warning: non-finite log_prob at iter {i+1}; stopping early.")
            break

        grad = torch.autograd.grad(lp, current, create_graph=False)[0]
        if not torch.isfinite(grad).all():
            print(f"Warning: non-finite gradient at iter {i+1}; stopping early.")
            break

        # Clip gradients to prevent explosion (default: clip to [-10, 10])
        if grad_clip is not None:
            grad_norm = grad.norm()
            if grad_norm > grad_clip:
                grad = grad * (grad_clip / grad_norm)
        
        # SGLD update: θ_{t+1} = θ_t + (ε/2) * ∇log p(θ|D) + N(0, ε)
        noise = torch.randn_like(current) * np.sqrt(eps)
        current = (current + 0.5 * eps * grad + noise).detach().requires_grad_(True)
        
        # Decay step size
        eps = max(eps * step_decay, min_step_size)
        
        # Collect samples after burn-in
        if i >= burn_in:
            samples.append(current.detach().clone())
        
        # Progress reporting
        if (i + 1) % 100 == 0:
            phase = "burn-in" if i < burn_in else "sampling"
            print(f"Iter {i+1:4d}/{total_iterations}: step_size = {eps:.2e}, phase = {phase}")
    
    print(f"SGLD completed. Collected {len(samples)} samples.")
    samples = torch.stack(samples)
    return samples, eps


# ============================================================
# Compute the Hessian using Gauss-Newton approximation
# For regression with Gaussian likelihood:
#   H ≈ J^T J / σ² + I / σ_prior²
# where J is the Jacobian of the network output w.r.t. parameters
# ============================================================

def compute_diagonal_hessian(model, x_branch, x_trunk, y, noise_std, prior_std, device, batch_size=20, sample_points_per_batch=200):
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
    params = [p for p in model.parameters() if p.requires_grad]
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
        n_outputs = test_pred.numel() // test_pred.shape[0]  # Total flattened output dimension per sample
    
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
        
        # Sample output indices from ALL output dimensions
        sample_indices = np.random.choice(n_outputs, min(sample_points_per_batch, n_outputs), replace=False)
        
        # Forward pass
        pred = model.forward(x_b_batch, x_t)  
        # Reshape to [batch_size, n_outputs] to ensure we can index scalar outputs
        batch_size_actual = pred.shape[0]
        pred = pred.reshape(batch_size_actual, -1)
        
        # Process each sample and sampled output point
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

def uqevaluation(num_test, test_data, model, method, hmc_samples=None, la_samples=None, sgld_samples=None, model_ensemble=None):

    # Define noise standard deviation (should match the value used in HMC sampling)
    noise_std = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_indices = np.random.choice(len(test_data["X_train"]), num_test, replace=False)
    eval_indices.sort()
    epoch_mcd = 100

    x_branch_eval = test_data['X_train'][eval_indices]
    x_trunk_eval = test_data['X_trunk']
    y_eval = test_data['Y_train'][eval_indices]

    print(f"Evaluating uncertainty on {num_test} test samples...")
    
    if method == 'hmc':
        # Compute predictions for each posterior sample
        print("Computing posterior predictions...")
        with torch.no_grad():
            preds_eval_list = []
            for idx, s in enumerate(hmc_samples): # pyright: ignore[reportArgumentType]
                unpack_params(model, s.to(device))
                x_b = torch.from_numpy(x_branch_eval).float().to(device)
                x_t = torch.from_numpy(x_trunk_eval).float().to(device)
                pred = model.predict(x_b, x_t)
                preds_eval_list.append(pred.cpu().numpy())
        preds_eval = np.stack(preds_eval_list)
    elif method == 'sgld':
        # Compute SGLD posterior predictions
        print("Computing SGLD posterior predictions...")
        preds_eval_list = []
        with torch.no_grad():
            for idx, s in enumerate(sgld_samples): # pyright: ignore[reportArgumentType]
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
            for idx, s in enumerate(la_samples): # pyright: ignore[reportArgumentType]
                unpack_params(model, s.to(device))
                x_b = torch.from_numpy(x_branch_eval).float().to(device)
                x_t = torch.from_numpy(x_trunk_eval).float().to(device)
                pred = model.predict(x_b, x_t)
                preds_eval_list.append(pred.cpu().numpy())
        preds_eval = np.stack(preds_eval_list)
    elif method == 'de':
        # Compute Deep Ensemble predictions
        print("Computing Deep Ensemble predictions...")
        preds_eval_list = []
        with torch.no_grad():
            for paths in model_ensemble:  # model is a list of paths for models.
                m = torch.load(paths, weights_only=False).to(device)
                x_b = torch.from_numpy(x_branch_eval).float().to(device)
                x_t = torch.from_numpy(x_trunk_eval).float().to(device)
                pred = m.predict(x_b, x_t)
                preds_eval_list.append(pred.cpu().numpy())
        preds_eval = np.stack(preds_eval_list)
    else:
        raise ValueError(f"Unknown UQ method: {method}")
    
    return uq_evaluation.compute_metric(preds_eval, noise_std, y_eval)
