import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import hamiltorch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append(".")
import uq_evaluation

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

def _flatten_param_list(params):
    """Flatten a list of Parameters, splitting complex tensors into real and imag parts."""
    flat_parts = []
    for p in params:
        p_detached = p.detach()
        if p_detached.is_complex():
            flat_parts.append(p_detached.real.contiguous().view(-1))
            flat_parts.append(p_detached.imag.contiguous().view(-1))
        else:
            flat_parts.append(p_detached.contiguous().view(-1))
    return torch.cat(flat_parts) if flat_parts else torch.tensor([], device=params[0].device if params else 'cpu')


def _unflatten_param_list(params, flat):
    """Restore a flat tensor into the provided Parameter list (order-sensitive)."""
    device = params[0].device if params else flat.device
    flat = flat.to(device)
    offset = 0
    for p in params:
        numel = p.numel()
        if p.is_complex():
            if offset + 2 * numel > flat.numel():
                raise ValueError(f"Parameter unpacking overflow: offset {offset}, numel {numel}, flat size {flat.numel()}")
            real_part = flat[offset:offset + numel].view(p.shape)
            offset += numel
            imag_part = flat[offset:offset + numel].view(p.shape)
            offset += numel
            p.data.copy_(torch.complex(real_part, imag_part))
        else:
            if offset + numel > flat.numel():
                raise ValueError(f"Parameter unpacking overflow: offset {offset}, numel {numel}, flat size {flat.numel()}")
            p.data.copy_(flat[offset:offset + numel].view(p.shape))
            offset += numel
    if offset != flat.numel():
        raise ValueError(f"Unused parameters in flat vector: consumed {offset}, provided {flat.numel()}")


def _param_list_size(params):
    """Compute flattened length for a list of Parameters (counts real+imag)."""
    total = 0
    for p in params:
        total += 2 * p.numel() if p.is_complex() else p.numel()
    return total


def pack_params(model):
    """
    Flatten all trainable model parameters into a single 1D real-valued tensor.
    Complex parameters are split into real and imaginary parts.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    return _flatten_param_list(trainable)


def pack_all_params(model):
    """Flatten all parameters (trainable or frozen). Useful for full-model checkpoints."""
    return _flatten_param_list(list(model.parameters()))


def unpack_params(model, flat):
    """
    Unflatten a 1D real-valued tensor back into trainable model parameters.
    Complex parameters are reconstructed from real and imaginary parts.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    expected_size = _param_list_size(trainable)
    if flat.numel() != expected_size:
        raise ValueError(f"Flat size mismatch for trainable params: expected {expected_size}, got {flat.numel()}")
    _unflatten_param_list(trainable, flat)


def unpack_all_params(model, flat):
    """Unflatten a flat vector into all parameters (trainable or frozen)."""
    params = list(model.parameters())
    expected_size = _param_list_size(params)
    if flat.numel() != expected_size:
        raise ValueError(f"Flat size mismatch for all params: expected {expected_size}, got {flat.numel()}")
    _unflatten_param_list(params, flat)

def get_total_param_size(model):
    """
    Get total size of flattened parameter vector (accounting for complex params).
    """
    return _param_list_size([p for p in model.parameters() if p.requires_grad])


def get_total_param_size_all(model):
    """Total flattened size including frozen parameters."""
    return _param_list_size(list(model.parameters()))

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


_HMC_SUBSET_ALIASES = {
    "all": "all",
    "full": "all",
    "full_layer": "all",
    "all_params": "all",
    "all_parameters": "all",
    "real_all": "real_all",
    "real": "real_all",
    "projectors": "projectors",
    "proj": "projectors",
    "head": "projectors",
    "last_layer": "last_layer",
    "last-layer": "last_layer",
    "last": "last_layer",
    "output_projector": "last_layer",
}


_LA_SUBSET_ALIASES = {
    "real_all": "real_all",
    "all": "real_all",
    "full": "real_all",
    "full_layer": "real_all",
    "all_params": "real_all",
    "all_parameters": "real_all",
    "projectors": "projectors",
    "proj": "projectors",
    "head": "projectors",
    "last_layer": "last_layer",
    "last-layer": "last_layer",
    "last": "last_layer",
    "output_projector": "last_layer",
}


def _normalize_hmc_subset_of_weights(subset_of_weights):
    key = str(subset_of_weights).strip().lower()
    if key not in _HMC_SUBSET_ALIASES:
        raise ValueError(
            f"Unsupported HMC subset_of_weights='{subset_of_weights}'. "
            "Use one of: 'all', 'real_all', 'projectors', 'last_layer'."
        )
    return _HMC_SUBSET_ALIASES[key]


def _normalize_la_subset_of_weights(subset_of_weights):
    key = str(subset_of_weights).strip().lower()
    if key not in _LA_SUBSET_ALIASES:
        raise ValueError(
            f"Unsupported Laplace subset_of_weights='{subset_of_weights}'. "
            "Use one of: 'real_all', 'projectors', 'last_layer'."
        )
    return _LA_SUBSET_ALIASES[key]


def _set_requires_grad_subset_fno(model, subset_key):
    """
    Configure trainable parameters for a selected FNO subset.

    Returns:
        mode: "all" (includes complex params) or "real_only"
    """
    for p in model.parameters():
        p.requires_grad = False

    if subset_key == "all":
        for p in model.parameters():
            p.requires_grad = True
        return "all"

    if subset_key == "real_all":
        for p in model.parameters():
            if not p.is_complex():
                p.requires_grad = True
        return "real_only"

    if subset_key == "projectors":
        if not hasattr(model, "input_projector") or not hasattr(model, "output_projector"):
            raise AttributeError("FNO model is missing input/output projectors.")
        for p in model.input_projector.parameters():
            p.requires_grad = True
        for p in model.output_projector.parameters():
            p.requires_grad = True
        return "real_only"

    if subset_key == "last_layer":
        if not hasattr(model, "output_projector"):
            raise AttributeError("FNO model has no output_projector.")
        for p in model.output_projector.parameters():
            p.requires_grad = True
        return "real_only"

    raise ValueError(f"Unknown subset key '{subset_key}'")


def _output_projector_params(model):
    """Return the Parameter list of the FNO output projector (last layer)."""
    if not hasattr(model, 'output_projector'):
        raise AttributeError("Model has no attribute 'output_projector'; cannot isolate last layer.")
    return list(model.output_projector.parameters())


def apply_sample_vector(model, flat):
    """
    Load a flattened parameter vector into the model while handling different sampling scopes.

    The sampler may produce vectors for:
      - only the trainable parameters (default behavior when layers are frozen),
      - only the output projector (last layer), or
      - the full model.

    The function matches the vector length to one of these cases and applies it accordingly.
    """
    device = next(model.parameters()).device
    flat = flat.to(device)

    full_size = get_total_param_size_all(model)
    trainable_size = get_total_param_size(model)
    last_layer_params = _output_projector_params(model)
    last_layer_size = _param_list_size(last_layer_params)
    flat_size = flat.numel()

    if flat_size == trainable_size:
        unpack_params(model, flat)
    elif flat_size == last_layer_size:
        # Only last layer was sampled. Keep other parameters fixed.
        _unflatten_param_list(last_layer_params, flat)
    elif flat_size == full_size:
        # Full-model sample vector (includes frozen params)
        unpack_all_params(model, flat)
    else:
        raise ValueError(
            f"Sample vector has length {flat_size}, which does not match last-layer ({last_layer_size}), "
            f"trainable ({trainable_size}), or full-model ({full_size}) parameter counts.")


# --- Helpers for working with real-valued subsets of the model ---

def _real_param_metadata(model):
    """Collect metadata for trainable real-valued parameters (skip complex weights)."""
    meta = []
    for name, p in model.named_parameters():
        if not p.requires_grad or p.is_complex():
            continue
        meta.append({
            "name": name,
            "param": p,
            "shape": p.shape,
            "dtype": p.dtype,
            "device": p.device,
            "numel": p.numel(),
        })
    return meta


def _real_param_size(meta):
    return sum(item["numel"] for item in meta)


def real_param_metadata(model):
    """Public helper to expose real-only trainable parameter metadata."""
    return _real_param_metadata(model)


def pack_real_params(meta):
    """Flatten only real-valued parameters described by metadata."""
    return _flatten_param_list([item["param"] for item in meta])


def build_full_vector_from_real_sample(model, real_meta, real_flat, base_state=None):
    """
    Reconstruct a full flattened parameter vector from a real-only sample.

    Args:
        model: FNO model.
        real_meta: Metadata list from real_param_metadata(model).
        real_flat: Flat tensor of real-valued parameters sampled by an algorithm.
        base_state: Optional reference state_dict to load before inserting the real sample.
    Returns:
        A full flattened vector (with complex weights untouched) ready for apply_sample_vector.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        if base_state is not None:
            model.load_state_dict(base_state, strict=False)
        _load_real_params_from_flat(real_meta, real_flat.to(device))
        return pack_all_params(model).detach().clone().cpu()


def _load_real_params_from_flat(meta, flat):
    """Copy a flat real-valued vector into the model parameters described by meta."""
    expected = _real_param_size(meta)
    if flat.numel() != expected:
        raise ValueError(f"Flat vector has size {flat.numel()}, expected {expected} for real parameters")

    offset = 0
    for item in meta:
        numel = item["numel"]
        tensor = flat[offset:offset + numel].view(item["shape"]).to(
            device=item["device"], dtype=item["dtype"])
        item["param"].data.copy_(tensor)
        offset += numel


def _functional_forward_real_subset(model, flat_params, X, real_meta):
    """
    Functional forward pass that swaps only the real-valued parameters defined in real_meta.
    Keeps complex spectral weights fixed while enabling gradients w.r.t. the sampled real subset.
    """
    if not real_meta:
        raise ValueError("No real-valued parameters found to substitute.")

    expected_size = _real_param_size(real_meta)
    if flat_params.numel() != expected_size:
        raise ValueError(
            f"Flat parameter vector size mismatch: expected {expected_size}, got {flat_params.numel()}")

    # Reconstruct tensors for the real parameters
    offset = 0
    new_tensors = []
    for item in real_meta:
        numel = item["numel"]
        tensor = flat_params[offset:offset + numel].view(item["shape"]).to(
            device=item["device"], dtype=item["dtype"])
        new_tensors.append(tensor)
        offset += numel

    backup = []
    try:
        # Swap real parameters with tensors tied to flat_params
        for item, new_tensor in zip(real_meta, new_tensors):
            atoms = item["name"].split('.')
            parent = model
            for atom in atoms[:-1]:
                parent = getattr(parent, atom)

            attr_name = atoms[-1]
            original = getattr(parent, attr_name)
            backup.append((parent, attr_name, original))

            delattr(parent, attr_name)
            setattr(parent, attr_name, new_tensor)

        return model(X)
    finally:
        # Restore original parameters
        for parent, attr_name, original in reversed(backup):
            if hasattr(parent, attr_name):
                delattr(parent, attr_name)
            setattr(parent, attr_name, original)



# ============================================================
# HMC (Hamiltonian Monte Carlo) for FNO
# ============================================================

def _functional_forward(model, flat_params, X):
    """
    Functional forward pass that creates gradient connections to flat_params.
    Safely swaps model parameters with tensors from flat_params.
    """
    # Check that flat_params has the correct size
    expected_size = get_total_param_size(model) # Now uses requires_grad
    if flat_params.numel() != expected_size:
        raise ValueError(f"Flat parameter vector size mismatch in _functional_forward: expected {expected_size}, got {flat_params.numel()}")
    
    # 1. Reconstruct parameter tensors from the flat vector
    offset = 0
    param_tensors = []
    
    for p in model.parameters():
        if not p.requires_grad:
            continue
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
        # Materialize the list of names BEFORE iterating.
        # Calling model.named_parameters() inside the loop causes the
        # "dictionary changed size" error because we delete attributes inside.
        param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        
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
    reduce_output_mean: bool = False,
    real_params_only: bool = False,
    real_meta=None):
    """
    Create log probability function for sampling.

    If ``real_params_only`` is True, only real-valued trainable parameters are used
    (complex weights in spectral layers are held fixed). Otherwise, all trainable
    parameters are included with complex parameters split into real/imag parts.
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

    if real_params_only:
        meta = real_meta if real_meta is not None else _real_param_metadata(model)
        if not meta:
            raise ValueError("No real-valued trainable parameters found for real-only log_prob.")
        real_size = _real_param_size(meta)

        def log_prob(flat_params):
            if flat_params.numel() != real_size:
                raise ValueError(f"Expected real parameter vector of size {real_size}, got {flat_params.numel()}")

            lp = -0.5 * (flat_params.pow(2).sum() / (prior_std ** 2))

            if batch_size is None or batch_size >= N:
                pred = _functional_forward_real_subset(model, flat_params, X, meta)
                resid = (y - pred).reshape(N, -1)
                resid2 = resid.pow(2).mean(dim=1).sum() if reduce_output_mean else resid.pow(2).sum()
                ll = -0.5 * (resid2 / (noise_std ** 2))
            else:
                indices = torch.randperm(N, device=device)[:batch_size]
                X_batch = X[indices]
                y_batch = y[indices]

                pred = _functional_forward_real_subset(model, flat_params, X_batch, meta)
                resid = (y_batch - pred).reshape(batch_size, -1)
                resid2 = resid.pow(2).mean(dim=1).sum()
                ll = -0.5 * (resid2 / (noise_std ** 2)) * (N / batch_size)

            return ll + lp

        return log_prob

    def log_prob(flat_params):
        # Log-prior (Gaussian) - all parameters (real + imaginary parts)
        lp = -0.5 * (flat_params.pow(2).sum() / (prior_std**2))

        if batch_size is None or batch_size >= N:
            pred = _functional_forward(model, flat_params, X)
            resid = (y - pred).reshape(N, -1)
            if reduce_output_mean:
                resid2 = resid.pow(2).mean(dim=1).sum()
            else:
                resid2 = resid.pow(2).sum()
            ll = -0.5 * (resid2 / (noise_std**2))
        else:
            indices = torch.randperm(N, device=device)[:batch_size]
            X_batch = X[indices]
            y_batch = y[indices]

            pred = _functional_forward(model, flat_params, X_batch)
            resid = (y_batch - pred).reshape(batch_size, -1)
            resid2 = resid.pow(2).mean(dim=1).sum()
            ll = -0.5 * (resid2 / (noise_std**2)) * (N / batch_size)

        return ll + lp

    return log_prob


def sgld(log_prob_fn, initial, step_size=5e-5, num_samples=2000, burn_in=2000,
         step_decay=0.9999, min_step_size=1e-7, grad_clip=100.0):
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

def hmc_nuts(model, X, y, num_samples=1000, burn_in=1000, step_size=1e-4, num_steps_per_sample=20, noise_std=0.2, prior_std=1.0):
    device = next(model.parameters()).device

    # Convert data to tensors on the correct device
    if isinstance(X, np.ndarray):
        X_tensor = torch.from_numpy(X).float().to(device)
    else:
        X_tensor = X.to(device)

    if isinstance(y, np.ndarray):
        y_tensor = torch.from_numpy(y).float().to(device)
    else:
        y_tensor = y.to(device)

    # Gather real-valued trainable parameters (Linear layers and Conv weights), skip complex spectral weights
    real_meta = _real_param_metadata(model)
    if not real_meta:
        raise ValueError("No real-valued trainable parameters found for HMC sampling.")

    params_flat = _flatten_param_list([item["param"] for item in real_meta]).to(device).requires_grad_(True)
    real_size = _real_param_size(real_meta)
    def log_prob_func(params):
        params = params.to(device)
        if params.numel() != real_size:
            raise ValueError(f"Expected real parameter vector of size {real_size}, got {params.numel()}")
        pred = _functional_forward_real_subset(model, params, X_tensor, real_meta)
        resid = (y_tensor - pred)
        ll = -0.5 * resid.pow(2).sum() / (noise_std ** 2)
        lp = -0.5 * params.pow(2).sum() / (prior_std ** 2)
        return ll + lp

    print("Running HMC (NUTS) on all real-valued layers using hamiltorch...")
    samples_real = hamiltorch.sample(
        log_prob_func=log_prob_func,
        params_init=params_flat,
        num_samples=num_samples,
        step_size=step_size,
        num_steps_per_sample=num_steps_per_sample,
        sampler=hamiltorch.Sampler.HMC_NUTS,
        burn=burn_in)

    full_samples = []
    print("Reconstructing full parameter samples (including complex weights held fixed)...")

    original_state = [item["param"].detach().clone() for item in real_meta]

    for s in samples_real:
        s = s.to(device)
        _load_real_params_from_flat(real_meta, s)
        full_s = pack_params(model).detach().clone()
        full_samples.append(full_s)

    # Restore original real-valued parameters
    for item, orig in zip(real_meta, original_state):
        item["param"].data.copy_(orig)

    full_samples = torch.stack(full_samples)
    return full_samples


def fit_hmc_torch(
    model,
    X,
    y,
    noise_std=0.2,
    subset_of_weights="projectors",
    prior_std=None,
    initial_step_size=1e-4,
    leapfrog_steps=20,
    num_samples=1000,
    burn_in=1000,
    random_seed=42,
    batch_size=None,
    reduce_output_mean=True,
):
    """
    Run HMC sampling for FNO and return full parameter vectors.

    Important note for FNO:
      - subset='all' samples all parameters, including complex Fourier weights
        represented via concatenated real/imag parts.
      - subset in {'real_all', 'projectors', 'last_layer'} samples only a real-valued
        subset and reconstructs full vectors with complex weights fixed at MAP values.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    hamiltorch.set_random_seed(random_seed)

    device = next(model.parameters()).device
    model_for_hmc = copy.deepcopy(model).to(device)
    base_state = {k: v.detach().clone() for k, v in model_for_hmc.state_dict().items()}

    subset_key = _normalize_hmc_subset_of_weights(subset_of_weights)
    mode = _set_requires_grad_subset_fno(model_for_hmc, subset_key)

    if mode == "all":
        flat0 = pack_params(model_for_hmc).to(device)
        empirical_prior_std = float(torch.std(flat0).detach().cpu().item()) if flat0.numel() > 1 else 1.0
        prior_std_eff = max(float(prior_std) if prior_std is not None else empirical_prior_std, 1e-12)
        log_prob = make_log_prob_fn(
            model_for_hmc,
            X,
            y,
            noise_std=noise_std,
            prior_std=prior_std_eff,
            batch_size=batch_size,
            reduce_output_mean=reduce_output_mean,
            real_params_only=False,
        )
        reconstruct_full = False
        real_meta = None
        print(
            "Preparing full-parameter HMC (complex-safe split real/imag): "
            f"n_trainable={flat0.numel()}, empirical_prior_std={empirical_prior_std:.3e}, "
            f"effective_prior_std={prior_std_eff:.3e}"
        )
    else:
        real_meta = _real_param_metadata(model_for_hmc)
        if not real_meta:
            raise ValueError("Selected HMC subset has zero real-valued trainable parameters.")
        flat0 = pack_real_params(real_meta).to(device)
        empirical_prior_std = float(torch.std(flat0).detach().cpu().item()) if flat0.numel() > 1 else 1.0
        prior_std_eff = max(float(prior_std) if prior_std is not None else empirical_prior_std, 1e-12)
        log_prob = make_log_prob_fn(
            model_for_hmc,
            X,
            y,
            noise_std=noise_std,
            prior_std=prior_std_eff,
            batch_size=batch_size,
            reduce_output_mean=reduce_output_mean,
            real_params_only=True,
            real_meta=real_meta,
        )
        reconstruct_full = True
        print(
            f"Preparing subset HMC ({subset_key}): "
            f"n_trainable_real={flat0.numel()}, empirical_prior_std={empirical_prior_std:.3e}, "
            f"effective_prior_std={prior_std_eff:.3e}"
        )

    sampled_list = hamiltorch.sample(
        log_prob_func=log_prob,
        params_init=flat0.requires_grad_(True),
        num_samples=num_samples,
        burn=burn_in,
        step_size=initial_step_size,
        num_steps_per_sample=leapfrog_steps,
        sampler=hamiltorch.Sampler.HMC_NUTS,
    )
    sampled = torch.stack(sampled_list)

    if not reconstruct_full:
        return sampled.detach().cpu()

    full_samples = []
    for s in sampled:
        full_samples.append(
            build_full_vector_from_real_sample(
                model_for_hmc,
                real_meta, # type: ignore[arg-type]
                s.detach(),
                base_state=base_state,
            )
        )
    return torch.stack(full_samples, dim=0)


def _collect_swag_snapshot(model, swag_state, max_rank):
    """Update SWAG running statistics with the current full-parameter snapshot."""
    w = pack_all_params(model).detach().cpu()

    if swag_state["n_models"] == 0:
        swag_state["mean"] = w.clone()
        swag_state["sq_mean"] = w.pow(2)
        swag_state["n_models"] = 1
        return

    n_old = swag_state["n_models"]
    n_new = n_old + 1
    mean_old = swag_state["mean"]
    sq_mean_old = swag_state["sq_mean"]

    mean_new = mean_old + (w - mean_old) / n_new
    sq_mean_new = sq_mean_old + (w.pow(2) - sq_mean_old) / n_new

    swag_state["mean"] = mean_new
    swag_state["sq_mean"] = sq_mean_new
    swag_state["n_models"] = n_new

    dev = w - mean_new
    swag_state["deviations"].append(dev)
    if len(swag_state["deviations"]) > max_rank:
        swag_state["deviations"].pop(0)


def fit_swag(
    model,
    train_data,
    swag_lr=5e-5,
    swag_epochs=50,
    batch_size=20,
    weight_decay=1e-4,
    momentum=0.9,
    collect_freq=1,
    start_collect_epoch=10,
    max_rank=20,
    random_seed=42,
    log_every=10,
):
    """
    Fit SWAG statistics for FNO by fine-tuning MAP weights with SGD and
    collecting full-parameter snapshots.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = next(model.parameters()).device

    x_train = train_data["X_train"]
    y_train = train_data["Y_train"]

    if isinstance(x_train, np.ndarray):
        x_train = torch.from_numpy(x_train).float()
    else:
        x_train = x_train.float().detach().cpu()
    if isinstance(y_train, np.ndarray):
        y_train = torch.from_numpy(y_train).float()
    else:
        y_train = y_train.float().detach().cpu()

    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=swag_lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    swag_state = {
        "mean": None,
        "sq_mean": None,
        "deviations": [],
        "n_models": 0,
        "max_rank": max_rank,
        "swag_lr": swag_lr,
    }

    print("Starting SWAG trajectory collection...")
    print(f"  Epochs: {swag_epochs}, Batch size: {batch_size}, LR: {swag_lr:.2e}")
    print(f"  Collect from epoch {start_collect_epoch} every {collect_freq} epoch(s), max rank: {max_rank}")

    nn.Module.train(model, True)
    for epoch in range(1, swag_epochs + 1):
        batch_losses = []
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model.forward(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        if epoch >= start_collect_epoch and ((epoch - start_collect_epoch) % collect_freq == 0):
            _collect_swag_snapshot(model, swag_state, max_rank=max_rank)

        if epoch == 1 or epoch == swag_epochs or epoch % log_every == 0:
            mean_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
            print(f"  Epoch {epoch:4d}/{swag_epochs}: train loss = {mean_loss:.4e}, collected = {swag_state['n_models']}")

    if swag_state["n_models"] == 0:
        raise RuntimeError("SWAG collected zero snapshots. Reduce start_collect_epoch or collect_freq.")

    if len(swag_state["deviations"]) > 0:
        swag_state["cov_mat_sqrt"] = torch.stack(swag_state["deviations"], dim=0)
    else:
        n_params = swag_state["mean"].numel()
        swag_state["cov_mat_sqrt"] = torch.empty((0, n_params), dtype=swag_state["mean"].dtype)
    del swag_state["deviations"]

    print(f"SWAG collection finished with {swag_state['n_models']} snapshots.")
    return swag_state


def sample_swag_posterior(
    swag_state,
    num_samples=30,
    scale=1.0,
    diag_only=False,
    var_clamp=1e-30,
    device=None,
):
    """
    Draw full-parameter samples from the SWAG Gaussian posterior approximation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = swag_state["mean"].to(device)
    sq_mean = swag_state["sq_mean"].to(device)
    cov_mat_sqrt = swag_state["cov_mat_sqrt"].to(device)

    diag_var = torch.clamp(sq_mean - mean.pow(2), min=var_clamp)
    diag_std = torch.sqrt(diag_var)

    samples = []
    diag_scale = np.sqrt(scale / 2.0)
    for _ in range(num_samples):
        z_diag = torch.randn_like(mean)
        sample = mean + diag_scale * diag_std * z_diag

        if (not diag_only) and cov_mat_sqrt.numel() > 0:
            k = cov_mat_sqrt.shape[0]
            if k > 1:
                z_low_rank = torch.randn(k, device=device)
                low_rank = cov_mat_sqrt.t().matmul(z_low_rank) / np.sqrt(k - 1.0)
                sample = sample + diag_scale * low_rank

        samples.append(sample.detach().cpu())

    return torch.stack(samples, dim=0)


class _FNOWrapper(nn.Module):
    """
    Adapter for laplace-torch: exposes FNO as a single-input regression model
    with flattened outputs and a PyTorch-compatible train(mode) toggle.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        pred = self.model.forward(x)
        return pred.reshape(pred.shape[0], -1)

    @staticmethod
    def _set_training_flag(module, mode):
        module.training = mode
        for child in module.children():
            _FNOWrapper._set_training_flag(child, mode)

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        self._set_training_flag(self.model, mode)
        return self


def fit_laplace_torch(
    model,
    X,
    y,
    batch_size=1,
    noise_std=0.2,
    prior_precision=1.0,
    subset_of_weights="projectors",
    hessian_structure="diag",
    random_seed=42,
):
    """
    Fit Laplace posterior for FNO with laplace-torch on real-valued subsets only.

    Supported subsets:
      - 'real_all': all non-complex trainable parameters
      - 'projectors': input + output projector layers
      - 'last_layer': output projector only
    """
    try:
        from laplace import Laplace
    except ImportError as exc:
        raise ImportError(
            "laplace-torch is not installed. Install with `pip install laplace-torch`."
        ) from exc

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    subset_key = _normalize_la_subset_of_weights(subset_of_weights)
    device = next(model.parameters()).device

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    else:
        X = X.float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    else:
        y = y.float()

    X = X.to(device)
    y = y.reshape(y.shape[0], -1).to(device)

    model_for_laplace = copy.deepcopy(model).to(device)
    base_state = {k: v.detach().clone() for k, v in model_for_laplace.state_dict().items()}

    _set_requires_grad_subset_fno(model_for_laplace, subset_key)
    real_meta = _real_param_metadata(model_for_laplace)
    if not real_meta:
        raise ValueError("Selected Laplace subset has zero real-valued trainable parameters.")

    wrapped_model = _FNOWrapper(model_for_laplace)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Fitting Laplace posterior with laplace-torch...")
    if isinstance(prior_precision, torch.Tensor):
        prior_precision_print = float(prior_precision.detach().cpu().item())
    else:
        prior_precision_print = float(prior_precision)
    print(f"  subset_of_weights={subset_of_weights} (effective={subset_key}), hessian_structure={hessian_structure}")
    print(f"  batch_size={batch_size}, sigma_noise={noise_std:.3e}, prior_precision={prior_precision_print:.3e}")

    la = Laplace(
        wrapped_model,
        likelihood="regression",
        subset_of_weights="all",
        hessian_structure=hessian_structure,
        sigma_noise=noise_std,
        prior_precision=prior_precision,
    )
    la.fit(loader)

    la._uq_model_ref = model_for_laplace
    la._uq_base_state = base_state
    la._uq_real_meta = real_meta
    la._uq_subset_key = subset_key
    return la


def sample_laplace_torch(la, num_samples=50):
    """
    Sample full-parameter vectors from a fitted FNO Laplace posterior.
    """
    samples = la.sample(num_samples)
    model_ref = la._uq_model_ref
    base_state = la._uq_base_state
    real_meta = la._uq_real_meta
    device = next(model_ref.parameters()).device

    full_samples = []
    for s in samples:
        full_samples.append(
            build_full_vector_from_real_sample(
                model_ref,
                real_meta,
                s.to(device),
                base_state=base_state,
            )
        )
    return torch.stack(full_samples, dim=0)


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
                             batch_size=20, sample_points_per_batch=200,
                             real_params_only: bool = True,
                             real_meta=None):
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

    if real_params_only:
        meta = real_meta if real_meta is not None else _real_param_metadata(model)
        if not meta:
            raise ValueError("No real-valued trainable parameters found for Hessian computation.")
        params = [item["param"] for item in meta]
        n_params = _real_param_size(meta)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
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

                    if real_params_only:
                        for item in meta:
                            p = item["param"]
                            numel = item["numel"]
                            if p.grad is not None:
                                grad_sq_sum[offset:offset + numel] = p.grad.view(-1).pow(2)
                            offset += numel
                    else:
                        for p in params:
                            if not p.requires_grad:
                                continue

                            numel = p.numel()
                            if p.grad is not None:
                                if p.is_complex():
                                    grad_sq_sum[offset:offset + numel] = p.grad.real.view(-1).pow(2)
                                    offset += numel
                                    grad_sq_sum[offset:offset + numel] = p.grad.imag.view(-1).pow(2)
                                    offset += numel
                                else:
                                    grad_sq_sum[offset:offset + numel] = p.grad.view(-1).pow(2)
                                    offset += numel
                            else:
                                offset += 2 * numel if p.is_complex() else numel

                    H_diag += grad_sq_sum * noise_var_inv * scale_factor
        
        # Explicitly delete tensors to free memory
        del pred, pred_flat, X_batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Progress reporting: print every 100 samples or at the end
        if batch_end % 100 == 0 or batch_end == n_samples:
            print(f"  Processed {batch_end}/{n_samples} samples")
    
    return H_diag


def _subsample_draws(draws, max_draws):
    """Uniformly subsample posterior draws while preserving endpoints."""
    if max_draws is None:
        return draws
    max_draws = int(max_draws)
    if max_draws <= 0:
        raise ValueError("max_posterior_samples must be positive when provided.")
    if len(draws) <= max_draws:
        return draws
    idx = np.linspace(0, len(draws) - 1, max_draws, dtype=int)
    if isinstance(draws, torch.Tensor):
        return draws[idx]
    return [draws[i] for i in idx]


def _resolve_eval_indices(eval_indices, dataset_size):
    """Support both explicit index arrays and legacy integer sample-count input."""
    if np.isscalar(eval_indices):
        num_test = int(eval_indices)
        if num_test <= 0:
            raise ValueError("num_test must be positive.")
        n_take = min(num_test, dataset_size)
        idx = np.random.choice(dataset_size, n_take, replace=False)
        idx.sort()
        return idx

    idx = np.asarray(eval_indices, dtype=int).reshape(-1)
    if idx.size == 0:
        raise ValueError("eval_indices must not be empty.")
    if np.any(idx < 0) or np.any(idx >= dataset_size):
        raise IndexError(
            f"eval_indices out of range: valid=[0, {dataset_size - 1}], got min={idx.min()}, max={idx.max()}."
        )
    return idx


def _take_rows(arr, idx):
    if isinstance(arr, np.ndarray):
        return arr[idx]
    if torch.is_tensor(arr):
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=arr.device)
        return arr.index_select(0, idx_t)
    arr_np = np.asarray(arr)
    return arr_np[idx]


def _to_tensor_float(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    if torch.is_tensor(x):
        return x.float().to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def uqevaluation(
    eval_indices,
    test_data,
    model,
    method,
    hmc_samples=None,
    sgld_samples=None,
    la_samples=None,
    swag_samples=None,
    model_ensemble=None,
    noise_std=0.2,
    epoch_mcd=100,
    max_posterior_samples=None,
    return_preds=False,
    flatten_output=False,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_total = len(test_data["X_train"])
    eval_indices = _resolve_eval_indices(eval_indices, n_total)

    X_eval = _take_rows(test_data["X_train"], eval_indices)
    y_eval = _take_rows(test_data["Y_train"], eval_indices)

    X_test_tensor = _to_tensor_float(X_eval, device)
    y_eval_np = _to_numpy(y_eval)
    predictions = []

    if method == 'hmc':
        if hmc_samples is None:
            raise ValueError("hmc_samples must be provided for method='hmc'.")
        draws = _subsample_draws(hmc_samples, max_posterior_samples)
        with torch.no_grad():
            for sample_params in draws:
                apply_sample_vector(model, sample_params)
                pred = model.predict(X_test_tensor).detach().cpu().numpy()
                predictions.append(pred)
    elif method == 'sgld':
        if sgld_samples is None:
            raise ValueError("sgld_samples must be provided for method='sgld'.")
        draws = _subsample_draws(sgld_samples, max_posterior_samples)
        with torch.no_grad():
            for sample_params in draws:
                apply_sample_vector(model, sample_params)
                pred = model.predict(X_test_tensor).detach().cpu().numpy()
                predictions.append(pred)
    elif method == 'mcd':
        with torch.no_grad():
            for _ in range(epoch_mcd):
                pred = model.forward(X_test_tensor).detach().cpu().numpy()
                predictions.append(pred)
    elif method == 'la':
        if la_samples is None:
            raise ValueError("la_samples must be provided for method='la'.")
        draws = _subsample_draws(la_samples, max_posterior_samples)
        with torch.no_grad():
            for sample_params in draws:
                apply_sample_vector(model, sample_params)
                pred = model.predict(X_test_tensor).detach().cpu().numpy()
                predictions.append(pred)
    elif method == 'swag':
        if swag_samples is None:
            raise ValueError("swag_samples must be provided for method='swag'.")
        draws = _subsample_draws(swag_samples, max_posterior_samples)
        with torch.no_grad():
            for sample_params in draws:
                apply_sample_vector(model, sample_params)
                pred = model.predict(X_test_tensor).detach().cpu().numpy()
                predictions.append(pred)
    elif method == 'de':
        if model_ensemble is None:
            raise ValueError("model_ensemble must be provided for method='de'.")
        with torch.no_grad():
            for path in model_ensemble:
                m = torch.load(path, weights_only=False).to(device)
                pred = m.predict(X_test_tensor).detach().cpu().numpy()
                predictions.append(pred)
    else:
        raise ValueError(f"Unknown method: {method}")

    predictions = np.stack(predictions)

    predictions_flat = predictions.reshape(predictions.shape[0], predictions.shape[1], -1)
    y_eval_flat = y_eval_np.reshape(y_eval_np.shape[0], -1)

    if return_preds:
        if flatten_output:
            return predictions_flat, y_eval_flat
        return predictions, y_eval_np

    return uq_evaluation.compute_metric(predictions_flat, noise_std, y_eval_flat)


def baseline(eval_indices, test_data, model):
    """
    Baseline deterministic RMSE using MAP model predictions.

    `eval_indices` can be an index array (preferred) or an int (legacy API).
    """
    n_total = len(test_data["X_train"])
    eval_indices = _resolve_eval_indices(eval_indices, n_total)

    X_eval = _take_rows(test_data["X_train"], eval_indices)
    y_eval = _take_rows(test_data["Y_train"], eval_indices)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_test_tensor = _to_tensor_float(X_eval, device)

    with torch.no_grad():
        pred = model.predict(X_test_tensor).detach().cpu().numpy()

    y_eval_np = _to_numpy(y_eval)
    errors = y_eval_np.reshape(y_eval_np.shape[0], -1) - pred.reshape(pred.shape[0], -1)
    rmse = np.sqrt(np.mean(errors ** 2))
    return float(rmse)
