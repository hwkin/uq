import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn as nn
src_path = "/../../"
import sys
sys.path.append(src_path + 'plotting/')
from field_plot import field_plot_grid as _field_plot_grid  # pyright: ignore[reportMissingImports]

# --- Utilities to flatten/unflatten model parameters ---
# NOTE: FNO models have complex-valued weights in SpectralConv layers.
# Standard parameters_to_vector doesn't handle complex tensors properly.
# These custom functions split complex parameters into real/imag parts.

def get_param_shapes(model):
    return [p.shape for p in model.parameters()]

def get_param_info(model):
    """
    Get parameter metadata including shapes, dtyp es, and whether they are complex.
    Returns a list of tuples: (shape, dtype, is_complex)
    """
    info = []
    for p in model.parameters():
        is_complex = p.is_complex()
        info.append((p.shape, p.dtype, is_complex))
    return info

def pack_params(model):
    """
    Flatten all model parameters into a single 1D real-valued tensor.
    Complex parameters are split into real and imaginary parts.
    """
    flat_parts = []
    for p in model.parameters():
        p_detached = p.detach()
        if p_detached.is_complex():
            # Split complex tensor into real and imaginary parts
            flat_parts.append(p_detached.real.contiguous().view(-1))
            flat_parts.append(p_detached.imag.contiguous().view(-1))
        else:
            flat_parts.append(p_detached.contiguous().view(-1))
    return torch.cat(flat_parts)

def unpack_params(model, flat):
    """
    Unflatten a 1D real-valued tensor back into model parameters.
    Complex parameters are reconstructed from real and imaginary parts.
    """
    device = next(model.parameters()).device
    flat = flat.to(device)
    
    # Check that flat has the correct size
    expected_size = get_total_param_size(model)
    if flat.numel() != expected_size:
        raise ValueError(f"Flat parameter vector size mismatch: expected {expected_size}, got {flat.numel()}")
    
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        if p.is_complex():
            # Complex parameters: read real and imag parts separately
            if offset + 2 * numel > flat.numel():
                raise ValueError(f"Parameter unpacking overflow: offset {offset}, numel {numel}, flat size {flat.numel()}")
            real_part = flat[offset:offset + numel].view(p.shape)
            offset += numel
            imag_part = flat[offset:offset + numel].view(p.shape)
            offset += numel
            # Reconstruct complex tensor
            p.data.copy_(torch.complex(real_part, imag_part))
        else:
            if offset + numel > flat.numel():
                raise ValueError(f"Parameter unpacking overflow: offset {offset}, numel {numel}, flat size {flat.numel()}")
            p.data.copy_(flat[offset:offset + numel].view(p.shape))
            offset += numel

def get_total_param_size(model):
    """
    Get total size of flattened parameter vector (accounting for complex params).
    """
    total = 0
    for p in model.parameters():
        if p.is_complex():
            total += 2 * p.numel()  # real + imag
        else:
            total += p.numel()
    return total

def _coerce_positive_scalar(value, name: str) -> float:
    """Coerce common scalar-like inputs (float/int/1-elem tuple/list/ndarray/tensor) to float."""
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise TypeError(f"{name} must be a scalar; got {type(value).__name__} of length {len(value)}")
        value = value[0]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise TypeError(f"{name} must be a scalar; got ndarray with shape {value.shape}")
        value = value.item()
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise TypeError(f"{name} must be a scalar; got tensor with shape {tuple(value.shape)}")
        value = value.detach().item()
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar; got {value}")
    return value



# ============================================================
# HMC (Hamiltonian Monte Carlo) for FNO
# ============================================================

def _functional_forward(model, flat_params, X):
    """
    Functional forward pass that creates gradient connections to flat_params.
    Safely swaps model parameters with tensors from flat_params.
    """
    # Check that flat_params has the correct size
    expected_size = get_total_param_size(model)
    if flat_params.numel() != expected_size:
        raise ValueError(f"Flat parameter vector size mismatch in _functional_forward: expected {expected_size}, got {flat_params.numel()}")
    
    # 1. Reconstruct parameter tensors from the flat vector
    offset = 0
    param_tensors = []
    
    for p in model.parameters():
        numel = p.numel()
        if p.is_complex():
            if offset + 2 * numel > flat_params.numel():
                raise ValueError(f"Parameter reconstruction overflow: offset {offset}, numel {numel}, flat size {flat_params.numel()}")
            real_part = flat_params[offset:offset + numel].view(p.shape)
            offset += numel
            imag_part = flat_params[offset:offset + numel].view(p.shape)
            offset += numel
            param_tensor = torch.complex(real_part, imag_part)
        else:
            if offset + numel > flat_params.numel():
                raise ValueError(f"Parameter reconstruction overflow: offset {offset}, numel {numel}, flat size {flat_params.numel()}")
            param_tensor = flat_params[offset:offset + numel].view(p.shape)
            offset += numel
        param_tensors.append(param_tensor)
    
    # 2. Swap model parameters with these new tensors
    backup = []
    
    try:
        # CRITICAL FIX: Materialize the list of names BEFORE iterating.
        # Calling model.named_parameters() inside the loop causes the
        # "dictionary changed size" error because we delete attributes inside.
        param_names = [n for n, _ in model.named_parameters()]
        
        for name, new_tensor in zip(param_names, param_tensors):
            # Traverse to the parent module
            atoms = name.split('.')
            parent = model
            for item in atoms[:-1]:
                parent = getattr(parent, item)
            
            attr_name = atoms[-1]
            
            # Save original and swap
            if hasattr(parent, attr_name):
                original = getattr(parent, attr_name)
                backup.append((parent, attr_name, original))
                
                # Delete the Parameter and set the Tensor
                delattr(parent, attr_name)
                setattr(parent, attr_name, new_tensor)
        
        # 3. Run the model forward pass
        return model(X)
        
    finally:
        # 4. Restore original parameters
        for parent, attr_name, original in reversed(backup):
            if hasattr(parent, attr_name):
                delattr(parent, attr_name)
            setattr(parent, attr_name, original)


def make_log_prob_fn(
    model,
    X,
    y,
    noise_std=0.01,
    prior_std=1.0,
    batch_size=None,
    reduce_output_mean: bool = False):
    """
    Create log probability function for HMC sampling for FNO.
    
    Args:
        model: FNO2D model
        X: Input data with shape (batch, nx, ny, 3) where 3 = (m, x, y)
        y: Target output with shape (batch, nx, ny, num_Y_components)
        noise_std: Observation noise standard deviation
        prior_std: Prior standard deviation for weights
        batch_size: Mini-batch size for memory-efficient evaluation. If None, uses all data.
                   Recommended for large datasets to avoid CUDA OOM errors.
        reduce_output_mean: If True, averages squared residuals over spatial/output points
            per sample (i.e., uses per-sample MSE then sums over samples). This keeps
            gradient magnitudes better-scaled for operator learning where each training
            example is a field with many correlated points.
    """
    noise_std = _coerce_positive_scalar(noise_std, "noise_std")
    prior_std = _coerce_positive_scalar(prior_std, "prior_std")

    device = next(model.parameters()).device
    
    # Convert to tensors if needed
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    
    X = X.to(device)
    y = y.to(device)
    
    N = X.shape[0]  # Total number of samples

    def log_prob(flat_params):
        # Log-prior (Gaussian) - always computed on all parameters
        lp = -0.5 * (flat_params.pow(2).sum() / (prior_std**2))
        
        if batch_size is None or batch_size >= N:
            # Use all data (original behavior)
            pred = _functional_forward(model, flat_params, X)
            resid = (y - pred).reshape(N, -1)
            # Log-likelihood (Gaussian)
            if reduce_output_mean:
                resid2 = resid.pow(2).mean(dim=1).sum()
            else:
                resid2 = resid.pow(2).sum()
            ll = -0.5 * (resid2 / (noise_std**2))
        else:
            # Mini-batch estimation with scaling
            # Randomly select a mini-batch
            indices = torch.randperm(N, device=device)[:batch_size]
            X_batch = X[indices]
            y_batch = y[indices]
            
            pred = _functional_forward(model, flat_params, X_batch)
            resid = (y_batch - pred).reshape(batch_size, -1)
            # Scale log-likelihood to account for mini-batch
            # Always use reduce_output_mean=True for mini-batch to avoid numerical instability
            # This computes per-sample MSE then sums, which is more stable than summing all points
            resid2 = resid.pow(2).mean(dim=1).sum()
            # Scale by (N/batch_size) to account for using only a subset of data
            ll = -0.5 * (resid2 / (noise_std**2)) * (N / batch_size)
        
        return ll + lp
    
    return log_prob


def sgld(log_prob_fn, initial, step_size=1e-5, num_samples=500, burn_in=100,
         step_decay=0.9999, min_step_size=1e-7, grad_clip=10.0):
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
    
    Returns:
        samples: List of parameter samples
        final_step_size: Final step size
    """
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
    return samples, eps



def hmc_adaptive(
    log_prob_fn,
    initial,
    target_accept=0.75,
    initial_step_size=1e-6,
    leapfrog_steps=10,
    num_samples=500,
    burn_in=100,
    adapt_steps=None,
    min_step_size=1e-8,
    max_step_size=1e-2,
    use_nuts=False,
    max_depth=8,
    divergence_threshold=100.0):
    """
    Hamiltonian Monte Carlo sampler with dual averaging step size adaptation for FNO.
    
    Args:
        log_prob_fn: Function that computes log probability given flat parameters
        initial: Initial parameter vector (requires_grad=True)
        target_accept: Target acceptance rate (0.65-0.80 is optimal for HMC)
        initial_step_size: Initial leapfrog step size
        leapfrog_steps: Number of leapfrog steps per iteration
        num_samples: Number of samples to collect after burn-in
        burn_in: Number of burn-in iterations
        adapt_steps: Number of steps to adapt step size (default: 80% of burn_in)
        min_step_size: Lower clamp for leapfrog step size during adaptation
        max_step_size: Upper clamp for leapfrog step size during adaptation
        use_nuts: If True, run No-U-Turn Sampler (NUTS) with slice sampling and tree doubling
        max_depth: Maximum tree depth for NUTS (trajectory length is ~2**max_depth steps)
        divergence_threshold: Threshold for declaring divergent transitions in NUTS
    
    Returns:
        samples: Tensor of shape (num_samples, dim)
        accept_rate: Final acceptance rate
        final_step_size: Adapted step size
        step_size_history: History of step sizes during adaptation
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
            if not torch.isfinite(lp):
                return q.detach(), p.detach(), lp.detach()
            grad = torch.autograd.grad(lp, q, create_graph=False)[0]
            p = p + 0.5 * eps * grad
            
            # Full step for position
            q = (q + eps * p).detach().requires_grad_(True)
            
            # Half step for momentum
            lp = log_prob_fn(q)
            if not torch.isfinite(lp):
                return q.detach(), p.detach(), lp.detach()
            grad = torch.autograd.grad(lp, q, create_graph=False)[0]
            p = p + 0.5 * eps * grad
        
        return q, p, lp.detach()

    # NUTS helpers ------------------------------------------------------
    def leapfrog_one(q, p, eps):
        """Single-step leapfrog used by NUTS (direction set by sign of eps)."""
        q = q.clone().detach().requires_grad_(True)
        lp = log_prob_fn(q)
        if not torch.isfinite(lp):
            # Force divergence handling upstream
            return q.detach(), p.detach(), lp.detach()
        grad = torch.autograd.grad(lp, q, create_graph=False)[0]
        p_half = p + 0.5 * eps * grad
        q_new = (q + eps * p_half).detach().requires_grad_(True)
        lp_new = log_prob_fn(q_new)
        if not torch.isfinite(lp_new):
            return q_new.detach(), p.detach(), lp_new.detach()
        grad_new = torch.autograd.grad(lp_new, q_new, create_graph=False)[0]
        p_new = p_half + 0.5 * eps * grad_new
        return q_new.detach(), p_new.detach(), lp_new.detach()

    def stop_criterion(q_minus, q_plus, p_minus, p_plus):
        """Check the No-U-Turn condition."""
        delta_q = (q_plus - q_minus).view(-1)
        return (torch.dot(delta_q, p_minus.view(-1)) >= 0) and (torch.dot(delta_q, p_plus.view(-1)) >= 0)

    def build_tree(q, p, log_u, v, j, eps, joint0):
        """Recursive tree building for NUTS."""
        if j == 0:
            # Base case: take one leapfrog step
            q_prime, p_prime, lp_prime = leapfrog_one(q, p, v * eps)
            joint = lp_prime - 0.5 * p_prime.pow(2).sum()
            n_prime = int((log_u <= joint).item())
            # flag divergence if energy error is too large (standard NUTS check)
            energy_error = (joint - log_u).abs()
            s_prime = int(energy_error < divergence_threshold)
            alpha_prime = min(1.0, torch.exp(joint - joint0).item()) if s_prime else 0.0
            if not s_prime:
                n_prime = 0  # discard this node on divergence
            return q_prime, p_prime, q_prime, p_prime, q_prime, lp_prime, n_prime, s_prime, alpha_prime, 1
        else:
            # Build the first half of the tree
            (q_minus, p_minus, q_plus, p_plus, q_proposal, lp_proposal,
             n_prime, s_prime, alpha_sum, n_alpha) = build_tree(q, p, log_u, v, j - 1, eps, joint0)

            if s_prime:
                # Build the second half
                if v == -1:
                    (q_minus, p_minus, _, _, q_right, lp_right,
                     n_right, s_right, alpha_right, n_alpha_right) = build_tree(q_minus, p_minus, log_u, v, j - 1, eps, joint0)
                else:
                    (_, _, q_plus, p_plus, q_right, lp_right,
                     n_right, s_right, alpha_right, n_alpha_right) = build_tree(q_plus, p_plus, log_u, v, j - 1, eps, joint0)

                # Decide whether to accept candidate from right subtree
                if s_right and torch.rand(()) < float(n_right) / float(max(n_prime + n_right, 1)):
                    q_proposal, lp_proposal = q_right, lp_right

                n_prime += n_right
                s_prime = s_right and stop_criterion(q_minus, q_plus, p_minus, p_plus)
                alpha_sum += alpha_right
                n_alpha += n_alpha_right

            return q_minus, p_minus, q_plus, p_plus, q_proposal, lp_proposal, n_prime, s_prime, alpha_sum, n_alpha

    accept_count = 0
    total_iterations = num_samples + burn_in
    step_size_history = []
    
    print(f"Starting adaptive HMC with target acceptance rate: {target_accept:.2%}")
    print(f"Adaptation will run for {adapt_steps} iterations")
    
    for i in range(total_iterations):
        # Sample momentum from standard normal
        p0 = torch.randn_like(current)

        if use_nuts:
            if max_depth < 1:
                raise ValueError("NUTS requires max_depth >= 1")
            # Slice variable
            current_joint = current_lp_val - 0.5 * p0.pow(2).sum()
            log_u = current_joint + torch.log(torch.rand((), device=current.device))

            q_minus = current
            q_plus = current
            p_minus = p0
            p_plus = p0
            j = 0
            n = 1
            s = 1
            q_prop = current
            lp_prop = current_lp_val
            alpha_sum = 0.0
            n_alpha = 0

            while s == 1 and j < max_depth:
                v = -1 if torch.rand(()) < 0.5 else 1
                if v == -1:
                    (q_minus, p_minus, _, _, q_candidate, lp_candidate,
                     n_prime, s_prime, alpha, n_alpha_prime) = build_tree(q_minus, p_minus, log_u, v, j, step_size, current_joint)
                else:
                    (_, _, q_plus, p_plus, q_candidate, lp_candidate,
                     n_prime, s_prime, alpha, n_alpha_prime) = build_tree(q_plus, p_plus, log_u, v, j, step_size, current_joint)

                if s_prime and torch.rand(()) < float(n_prime) / float(max(n, 1)):
                    q_prop, lp_prop = q_candidate, lp_candidate

                n += n_prime
                s = s_prime and stop_criterion(q_minus, q_plus, p_minus, p_plus)
                alpha_sum += alpha
                n_alpha += n_alpha_prime
                j += 1

            accept_prob = alpha_sum / max(n_alpha, 1)
            if np.isnan(accept_prob):
                accept_prob = 0.0
            if torch.rand(()) < accept_prob:
                current = q_prop.detach().requires_grad_(True)
                current_lp_val = lp_prop
                accept_count += 1
        else:
            # Standard fixed-step HMC
            q_prop, p_prop, lp_prop = leapfrog(current, p0, step_size)

            # Compute Hamiltonians (H = -log_prob + kinetic energy)
            current_H = -current_lp_val + 0.5 * p0.pow(2).sum()
            prop_H = -lp_prop + 0.5 * p_prop.pow(2).sum()

            # Compute acceptance probability stably; reject if non-finite
            log_accept_prob = (current_H - prop_H).detach()
            if not torch.isfinite(log_accept_prob):
                accept_prob = 0.0
            else:
                accept_prob = float(torch.exp(torch.clamp(log_accept_prob, max=0.0)).item())

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
            # Clamp step size to avoid exploding values when acceptance is high
            step_size = float(np.clip(np.exp(log_eps), min_step_size, max_step_size))
            log_eps = np.log(step_size)
            
            # Update averaged step size
            m_power = m ** (-kappa)
            log_eps_bar = m_power * log_eps + (1 - m_power) * log_eps_bar
            
        elif i == adapt_steps:
            # Fix step size to averaged value after adaptation
            step_size = float(np.clip(np.exp(log_eps_bar), min_step_size, max_step_size))
            log_eps_bar = np.log(step_size)
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
    
    # Check if we have any samples before stacking
    if len(samples) == 0:
        raise ValueError(f"No samples collected. Total iterations: {total_iterations}, Accept count: {accept_count}")
    
    return torch.stack(samples), final_accept_rate, step_size, step_size_history


def inject_dropout(model, target_layer_type=nn.Linear, dropout_rate=0.1):
    """
    Recursively adds a Dropout layer after every occurrence of `target_layer_type`.
    For FNO models, also handles FNO2DLayer and Conv2d layers.
    """
    # Import FNO2DLayer if available
    try:
        from torch_fno2dlayer import FNO2DLayer
    except ImportError:
        FNO2DLayer = None
    
    for name, child in model.named_children():
        # Special handling for ModuleList (used in FNO models for fno_layers)
        if isinstance(child, nn.ModuleList):
            # Process each element in the ModuleList
            for i, module in enumerate(child):
                # Skip if it's already a Dropout layer
                if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                    continue
                # Handle FNO2DLayer
                if FNO2DLayer is not None and isinstance(module, FNO2DLayer):
                    # Replace with Sequential: [FNO2DLayer, Dropout2d]
                    child[i] = nn.Sequential(module, nn.Dropout2d(dropout_rate))
                # Handle other layer types recursively
                else:
                    inject_dropout(module, target_layer_type, dropout_rate)
        # If the child is the target type (e.g., Linear)
        elif isinstance(child, target_layer_type):
            # Create a new Sequential container: [Original Layer, Dropout]
            new_layer = nn.Sequential(
                child, 
                nn.Dropout(dropout_rate)
            )
            # Replace the old child with the new wrapper
            setattr(model, name, new_layer)
        # Handle FNO2DLayer (custom layer in FNO models)
        elif FNO2DLayer is not None and isinstance(child, FNO2DLayer):
            # Use Dropout2d for 2D spatial data
            new_layer = nn.Sequential(
                child,
                nn.Dropout2d(dropout_rate)
            )
            setattr(model, name, new_layer)
        # Handle Conv2d layers (used in FNO2DLayer)
        elif isinstance(child, nn.Conv2d):
            # Use Dropout2d for 2D convolutional layers
            new_layer = nn.Sequential(
                child,
                nn.Dropout2d(dropout_rate)
            )
            setattr(model, name, new_layer)
        # If the child is a container (Sequential, ModuleList, or custom block), recurse
        else:
            inject_dropout(child, target_layer_type, dropout_rate)

# ============================================================
# Laplace Approximation for FNO
# ============================================================

def compute_diagonal_hessian(model, X, y, noise_std, prior_std, device, 
                             batch_size=20, sample_points_per_batch=50):
    """
    Compute diagonal approximation of the Hessian using the Gauss-Newton method for FNO.
    This is memory efficient: it processes one sample at a time and only samples
    a subset of output points.
    
    H_diag ≈ sum_i (∂f/∂θ)² / σ² + 1/σ_prior²
    
    NOTE: We use model.forward() instead of model.predict() because predict()
    wraps the forward pass in torch.no_grad(), which disables gradient computation.
    
    For complex parameters (FNO spectral weights), gradients are split into
    real and imaginary parts to match the flattened parameter representation.
    
    Args:
        model: FNO2D model
        X: Input data with shape (batch, nx, ny, 3)
        y: Target output with shape (batch, nx, ny, num_Y_components)
        noise_std: Observation noise standard deviation
        prior_std: Prior standard deviation for weights
        device: torch device
        batch_size: Number of samples to process at once
        sample_points_per_batch: Number of output points to sample per batch
    """
    noise_std = _coerce_positive_scalar(noise_std, "noise_std")
    prior_std = _coerce_positive_scalar(prior_std, "prior_std")
    if device is None:
        device = next(model.parameters()).device

    params = list(model.parameters())
    # Total params accounting for complex (real + imag parts)
    n_params = get_total_param_size(model)
    
    # Convert inputs to tensors once
    X_tensor = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X.clone()
    y_tensor = torch.from_numpy(y).float() if isinstance(y, np.ndarray) else y.clone()
    
    n_samples = X_tensor.shape[0]
    nx, ny = X_tensor.shape[1], X_tensor.shape[2]
    n_points = nx * ny
    
    # Initialize diagonal Hessian with prior term
    H_diag = torch.ones(n_params, device=device) / (prior_std ** 2)
    
    # Scale factor to account for subsampling output points
    scale_factor = n_points / sample_points_per_batch
    noise_var_inv = 1.0 / (noise_std ** 2)
    
    # Process samples in batches
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        X_batch = X_tensor[i:batch_end].to(device)
        
        # Sample output point indices (flattened)
        n_sample = min(sample_points_per_batch, n_points)
        if n_sample <= 0:
            raise ValueError(f"Cannot sample from {n_points} points with sample_points_per_batch={sample_points_per_batch}")
        sample_indices = np.random.choice(n_points, n_sample, replace=False)
        
        # Forward pass - use forward() to enable gradient computation
        pred = model.forward(X_batch)  # (batch_size, nx, ny, num_Y_components)
        
        # Flatten spatial dimensions for easier indexing
        pred_flat = pred.reshape(pred.shape[0], -1)  # (batch_size, nx*ny*num_Y_components)
        
        # Process each sample and sampled output point
        batch_size_actual = pred.shape[0]
        num_Y_components = pred.shape[-1]
        
        for j in range(batch_size_actual):
            for idx, k in enumerate(sample_indices):
                for c in range(num_Y_components):
                    # Zero gradients before backward
                    model.zero_grad()
                    
                    # Index into flattened prediction
                    flat_idx = k * num_Y_components + c
                    
                    # Compute gradient for this specific output
                    # Only release graph on the very last backward pass for this batch
                    is_last_in_batch = (j == batch_size_actual - 1) and (idx == len(sample_indices) - 1) and (c == num_Y_components - 1)
                    pred_flat[j, flat_idx].backward(retain_graph=not is_last_in_batch)
                    
                    # Accumulate squared gradients (diagonal Hessian approximation)
                    # Handle complex parameters by splitting into real/imag
                    grad_sq_sum = torch.zeros(n_params, device=device)
                    offset = 0
                    for p in params:
                        numel = p.numel()
                        if p.grad is not None:
                            if p.is_complex():
                                # Complex gradient: split into real and imag parts
                                grad_sq_sum[offset:offset + numel] = p.grad.real.view(-1).pow(2)
                                offset += numel
                                grad_sq_sum[offset:offset + numel] = p.grad.imag.view(-1).pow(2)
                                offset += numel
                            else:
                                grad_sq_sum[offset:offset + numel] = p.grad.view(-1).pow(2)
                                offset += numel
                        else:
                            if p.is_complex():
                                offset += 2 * numel
                            else:
                                offset += numel
                    
                    H_diag += grad_sq_sum * noise_var_inv * scale_factor
        
        # Explicitly delete tensors to free memory
        del pred, pred_flat, X_batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Progress reporting: print every 100 samples or at the end
        if batch_end % 100 == 0 or batch_end == n_samples:
            print(f"  Processed {batch_end}/{n_samples} samples")
    
    return H_diag


def uqevaluation(num_test, test_data, model, method, hmc_samples=None, sgld_samples=None, la_samples=None):
    aleatoric_std=0.05
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_indices = np.random.choice(len(test_data["X_train"]), num_test, replace=False)
    eval_indices.sort()
    epoch_mcd = 100

    if torch.is_tensor(test_data["X_train"]):
        X_test_tensor = test_data["X_train"].clone().detach().to(device)
        X_test_tensor = X_test_tensor[eval_indices]
    else:
        X_test_tensor = torch.from_numpy(test_data["X_train"]).float().to(device)
        X_test_tensor = X_test_tensor[eval_indices]
    y_eval = test_data['Y_train'].detach().cpu().numpy()
    y_eval = y_eval.reshape(y_eval.shape[0], -1)
    y_eval = y_eval[eval_indices]

    if method == 'hmc':
        predictions = []
        for i in range(hmc_samples.shape[0]): # pyright: ignore[reportOptionalMemberAccess]
            sample_params = hmc_samples[i] # pyright: ignore[reportOptionalSubscript]
            unpack_params(model, sample_params)
            with torch.no_grad():
                pred = model(X_test_tensor).detach().cpu().numpy()
                pred = pred.reshape(pred.shape[0], -1)
                predictions.append(pred)
        predictions = np.array(predictions)

    elif method == 'sgld':
        pred_batch_size = 20  # Adjust based on available GPU memory
        # Get predictions from each SGLD sample using mini-batches
        predictions = []
        # Handle both tensor and list of tensors
        for i in range(sgld_samples.shape[0]): # pyright: ignore[reportOptionalMemberAccess]
            sample_params = sgld_samples[i] # pyright: ignore[reportOptionalSubscript]
            unpack_params(model, sample_params)
            # Process test data in mini-batches
            sample_preds = []
            num_test_samples = X_test_tensor.shape[0]
            with torch.no_grad():
                for start_idx in range(0, num_test_samples, pred_batch_size):
                    end_idx = min(start_idx + pred_batch_size, num_test_samples)
                    X_batch = X_test_tensor[start_idx:end_idx]
                    pred_batch = model(X_batch).detach().cpu().numpy()
                    pred_batch = pred_batch.reshape(pred_batch.shape[0], -1)
                    sample_preds.append(pred_batch)
                    # Clear GPU cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            # Concatenate all batches for this sample
            predictions.append(np.concatenate(sample_preds, axis=0))
        predictions = np.array(predictions)

    elif method == 'mcd':
        predictions = []
        print("Running MC Dropout sampling...")
        for i in range(epoch_mcd):
            with torch.no_grad():
                pred = model(X_test_tensor).detach().cpu().numpy()
                pred = pred.reshape(pred.shape[0], -1)
                predictions.append(pred)
        predictions = np.array(predictions)

    elif method == 'la':
        predictions=[]
        for i in range(la_samples.shape[0]): # pyright: ignore[reportOptionalMemberAccess]
            sample_params = la_samples[i] # pyright: ignore[reportOptionalSubscript]
            unpack_params(model, sample_params)
            with torch.no_grad():
                pred = model(X_test_tensor).detach().cpu().numpy()
                pred = pred.reshape(pred.shape[0], -1)
                predictions.append(pred)
        predictions = np.array(predictions)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute statistics
    mean_eval = np.mean(predictions, axis=0)  # (N_test, ...)
    epistemic_std_eval = np.std(predictions, axis=0)    # Epistemic uncertainty
    epistemic_var_eval = epistemic_std_eval ** 2
    aleatoric_var_eval = aleatoric_std ** 2
    total_var_eval = epistemic_var_eval + aleatoric_var_eval
    total_std_eval = np.sqrt(total_var_eval)
    sample_std = np.mean(epistemic_std_eval, axis=1)
    
    # PREDICTION ERROR
    errors = y_eval - mean_eval
    squared_errors = errors ** 2
    rmse = np.sqrt(np.mean(squared_errors))

    # CALIBRATION
    z_scores = np.abs(errors) / total_std_eval

    coverage_1sigma = np.mean(z_scores <= 1.0)
    coverage_2sigma = np.mean(z_scores <= 2.0)
    coverage_3sigma = np.mean(z_scores <= 3.0)

    # SHARPNESS
    num_sigma = 2.0
    widths = 2 * num_sigma * total_std_eval
    mpiw = np.mean(widths)

    # NEGATIVE LOG-LIKELIHOOD
    total_var_safe = np.maximum(total_var_eval, 1e-12)
    nll = 0.5 * np.mean(np.log(2 * np.pi * total_var_safe) + squared_errors / total_var_safe)
    
    # Return metrics dict and computed values for compatibility
    return sample_std, np.array([rmse, coverage_1sigma, coverage_2sigma, coverage_3sigma, mpiw, nll, np.mean(total_std_eval)])

def comparison_uq(result1,result2,result3,result4):
    print("="*70)
    print("COMPARISON: HMC vs SGLD vs MC Dropout vs Laplace Approximation")
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
        'SGLD': [
            f'{result2[0]:.6f}',
            f'{result2[1]*100:.1f}',
            f'{result2[2]*100:.1f}',
            f'{result2[3]*100:.1f}',
            f'{result2[4]:.4f}',
            f'{result2[5]:.4f}'
        ],
        'MC Dropout': [
            f'{result3[0]:.6f}',
            f'{result3[1]*100:.1f}',
            f'{result3[2]*100:.1f}',
            f'{result3[3]*100:.1f}',
            f'{result3[4]:.4f}',
            f'{result3[5]:.4f}'
        ],
        'Laplace': [
            f'{result4[0]:.6f}',
            f'{result4[1]*100:.1f}',
            f'{result4[2]*100:.1f}',
            f'{result4[3]*100:.1f}',
            f'{result4[4]:.4f}',
            f'{result4[5]:.4f}'
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
    print("\n{:<25} {:>12} {:>12} {:>12} {:>12} {:>10}".format('Metric', 'HMC', 'SSGLD', 'MC Dropout', 'Laplace', 'Ideal'))
    print("-" * 85)
    for i in range(len(comparison_data['Metric'])):
        print("{:<25} {:>12} {:>12} {:>12} {:>12} {:>10}".format(
            comparison_data['Metric'][i],
            comparison_data['HMC'][i],
            comparison_data['SGLD'][i],
            comparison_data['MC Dropout'][i],
            comparison_data['Laplace'][i],
            comparison_data['Ideal'][i]
        ))

def run_regression_shift(method, levels, results):
    stats = {m: {'rmse': [], 'mpiw':[], 'nll': [], 'unc': [], 'cov': []} for m in method}

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