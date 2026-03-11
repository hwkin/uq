from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
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


_SUBSET_OF_WEIGHTS_ALIASES = {
    "all": "all",
    "full": "all",
    "full_layer": "all",
    "all_params": "all",
    "all_parameters": "all",
    "last_layer": "last_layer",
    "last-layer": "last_layer",
    "last": "last_layer",
}


def _normalize_subset_of_weights(subset_of_weights):
    """
    Normalize user-provided subset names to {'all', 'last_layer'}.

    Accepted aliases:
      - full model: all, full, full_layer, all_params, all_parameters
      - last layer: last_layer, last-layer, last
    """
    key = str(subset_of_weights).strip().lower()
    if key not in _SUBSET_OF_WEIGHTS_ALIASES:
        raise ValueError(
            f"Unsupported subset_of_weights='{subset_of_weights}'. "
            "Use 'last_layer' or 'full_layer' (alias: 'all')."
        )
    return _SUBSET_OF_WEIGHTS_ALIASES[key]

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

def fit_hmc_torch(
    model,
    x_branch,
    x_trunk,
    y,
    noise_std=0.2,
    subset_of_weights="last_layer",
    prior_std=None,
    initial_step_size=1e-4,
    leapfrog_steps=20,
    num_samples=1000,
    burn_in=1000,
    random_seed=42,
):
    """
    Run HMC sampling for DeepONet and return full parameter vectors.

    Args:
        model: Trained DeepONet model.
        x_branch: Branch inputs used for log-likelihood.
        x_trunk: Trunk coordinates.
        y: Targets.
        noise_std: Observation noise std for Gaussian likelihood.
        subset_of_weights: 'last_layer' (default) or 'full_layer'/'all' for
            full-parameter HMC.
        prior_std: Optional Gaussian prior std. If None, inferred from current
            sampled parameter subset.
        initial_step_size: HMC step size.
        leapfrog_steps: Number of leapfrog steps per HMC sample.
        num_samples: Number of posterior samples to draw.
        burn_in: Number of burn-in iterations.
        random_seed: Random seed.

    Returns:
        Tensor of shape [num_samples, n_full_params] on CPU.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = next(model.parameters()).device

    # Work on a copy so caller model parameters/flags remain unchanged.
    model_for_hmc = copy.deepcopy(model).to(device)
    base_state = {k: v.detach().clone() for k, v in model_for_hmc.state_dict().items()}
    all_params = list(model_for_hmc.parameters())

    subset_key = _normalize_subset_of_weights(subset_of_weights)
    if subset_key == "all":
        for p in model_for_hmc.parameters():
            p.requires_grad = True
        flat0 = pack_params(model_for_hmc).to(device)
        empirical_prior_std = float(torch.std(flat0).detach().cpu().item()) if flat0.numel() > 1 else 1.0
        prior_std_eff = max(float(prior_std) if prior_std is not None else empirical_prior_std, 1e-12)
        reconstruct_full = False
        print(
            "Preparing full-parameter HMC: "
            f"n_trainable={flat0.numel()}, empirical_prior_std={empirical_prior_std:.3e}, "
            f"effective_prior_std={prior_std_eff:.3e}"
        )
    else:
        model_for_hmc, flat0, empirical_prior_std = freezelayer(model_for_hmc, device)
        prior_std_eff = max(float(prior_std) if prior_std is not None else float(empirical_prior_std), 1e-12)
        reconstruct_full = True
        print(
            "Preparing last-layer HMC: "
            f"n_trainable={flat0.numel()}, empirical_prior_std={float(empirical_prior_std):.3e}, "
            f"effective_prior_std={prior_std_eff:.3e}"
        )

    log_prob = make_log_prob_fn(
        model_for_hmc,
        x_branch,
        x_trunk,
        y,
        noise_std=noise_std,
        prior_std=prior_std_eff,
    )

    sampled = hmc_nuts(
        log_prob,
        flat0.requires_grad_(True),
        initial_step_size=initial_step_size,
        leapfrog_steps=leapfrog_steps,
        num_samples=num_samples,
        burn_in=burn_in,
        random_seed=random_seed,
    )

    # For last-layer sampling, reconstruct full vectors for downstream loading.
    if reconstruct_full:
        full_samples = []
        for s in sampled:
            full_samples.append(build_full_vector(model_for_hmc, base_state, all_params, s.detach()))
        return torch.stack(full_samples, dim=0)

    return sampled.detach().cpu()

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


# --- SWA-Gaussian (SWAG) ---
def _collect_swag_snapshot(model, swag_state, max_rank):
    """
    Update SWAG running statistics with the current full-parameter snapshot.

    We keep:
      - running mean of parameters
      - running second moment for diagonal covariance
      - a low-rank matrix of centered deviations (SWAG covariance factor)
    """
    w = parameters_to_vector([p.detach().cpu() for p in model.parameters()])

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

    # Deviation w.r.t. updated mean (same approximation style used in SWAG codebases)
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
    Fit SWAG statistics by fine-tuning a trained MAP model with SGD and
    collecting parameter snapshots.

    Args:
        model: Trained DeepONet model (initialized at MAP solution).
        train_data: dict with 'X_train', 'X_trunk', 'Y_train'.
        swag_lr: SGD learning rate for SWAG trajectory collection.
        swag_epochs: Number of SWAG fine-tuning epochs.
        batch_size: Mini-batch size.
        weight_decay: SGD weight decay.
        momentum: SGD momentum.
        collect_freq: Collect one model every `collect_freq` epochs.
        start_collect_epoch: First epoch index (1-based) to start collecting.
        max_rank: Maximum number of low-rank deviation vectors to keep.
        random_seed: Random seed for reproducibility.
        log_every: Print training progress every `log_every` epochs.

    Returns:
        swag_state: dict containing running moments and covariance factor.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = next(model.parameters()).device

    x_branch = train_data["X_train"]
    y_train = train_data["Y_train"]
    x_trunk = train_data["X_trunk"]

    if isinstance(x_branch, np.ndarray):
        x_branch = torch.from_numpy(x_branch).float()
    else:
        x_branch = x_branch.float().detach().cpu()
    if isinstance(y_train, np.ndarray):
        y_train = torch.from_numpy(y_train).float()
    else:
        y_train = y_train.float().detach().cpu()

    if isinstance(x_trunk, np.ndarray):
        x_trunk = torch.from_numpy(x_trunk).float().to(device)
    else:
        x_trunk = x_trunk.float().to(device)

    dataset = TensorDataset(x_branch, y_train)
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
            pred = model.forward(xb, x_trunk)
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
        # No low-rank component if only one snapshot was collected
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
    Draw parameter samples from the SWAG Gaussian posterior approximation.

    Args:
        swag_state: Output dictionary from `fit_swag`.
        num_samples: Number of parameter samples.
        scale: Posterior covariance scaling.
        diag_only: If True, ignore low-rank covariance term.
        var_clamp: Minimum diagonal variance for numerical stability.
        device: Sampling device. Defaults to GPU if available, else CPU.

    Returns:
        samples: Tensor with shape [num_samples, n_params] on CPU.
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


# --- Laplace Approximation via laplace-torch ---
class _DeepONetBranchWrapper(nn.Module):
    """
    Adapter so laplace-torch can see a one-input model.
    It treats trunk coordinates as fixed and only exposes x_branch as input.
    """

    def __init__(self, model, x_trunk):
        super().__init__()
        self.model = model
        self.register_buffer("x_trunk", x_trunk)

    def forward(self, x_branch):
        return self.model.forward(x_branch, self.x_trunk)

    @staticmethod
    def _set_training_flag(module, mode):
        """
        Recursively set `.training` without calling `module.train(mode)`.
        DeepONet overloads `train(...)` for optimization, so the standard
        PyTorch mode toggle path is not available on that class.
        """
        module.training = mode
        for child in module.children():
            _DeepONetBranchWrapper._set_training_flag(child, mode)

    def train(self, mode=True):
        """
        PyTorch-compatible mode toggle used by laplace-torch.
        This intentionally bypasses `self.model.train(...)` because DeepONet's
        `train` method is overloaded with a different signature.
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")

        self.training = mode
        self._set_training_flag(self.model, mode)
        return self


class _ManualDiagLaplace:
    """
    Lightweight diagonal Laplace object with a `sample(num_samples)` API
    compatible with `sample_laplace_torch`.
    """

    def __init__(self, mean, posterior_std):
        self.mean = mean
        self.posterior_std = posterior_std

    def sample(self, num_samples):
        eps = torch.randn(
            (num_samples, self.mean.numel()),
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean.unsqueeze(0) + eps * self.posterior_std.unsqueeze(0)


def fit_laplace_torch(
    model,
    x_branch,
    x_trunk,
    y,
    batch_size=20,
    noise_std=0.2,
    prior_precision=1.0,
    subset_of_weights="all",
    hessian_structure="diag",
    random_seed=42,
    manual_sample_points_per_batch=512,
    manual_max_batch_size=10,
):
    """
    Fit a Laplace posterior using laplace-torch.

    Args:
        model: Trained DeepONet model.
        x_branch: Branch inputs (numpy array or tensor), shape [N, ...].
        x_trunk: Trunk coordinates (numpy array or tensor), shared across samples.
        y: Targets (numpy array or tensor), shape [N, ...].
        batch_size: Batch size for Hessian accumulation.
        noise_std: Observation noise std used by regression Laplace likelihood.
        prior_precision: Prior precision (inverse variance) for weights.
        subset_of_weights: Which parameter subset to fit.
            Use 'last_layer' or 'full_layer'/'all'.
        hessian_structure: laplace-torch Hessian structure, e.g., 'diag' or 'kron'.
        random_seed: Random seed.
        manual_sample_points_per_batch: Number of output dimensions sampled per
            batch in the manual diagonal fallback for DeepONet last-layer Laplace.
        manual_max_batch_size: Upper bound on batch size used by the manual
            diagonal fallback for memory control.

    Returns:
        la: Fitted laplace-torch object.
    """
    try:
        from laplace import Laplace
    except ImportError as exc:
        raise ImportError(
            "laplace-torch is not installed. Install with `pip install laplace-torch`."
        ) from exc

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    subset_of_weights = _normalize_subset_of_weights(subset_of_weights)

    device = next(model.parameters()).device

    if isinstance(x_branch, np.ndarray):
        x_branch = torch.from_numpy(x_branch).float()
    else:
        x_branch = x_branch.float()
    if isinstance(x_trunk, np.ndarray):
        x_trunk = torch.from_numpy(x_trunk).float()
    else:
        x_trunk = x_trunk.float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    else:
        y = y.float()

    # Keep data on same device as model; laplace-torch does not auto-move batch tensors.
    x_branch = x_branch.to(device)
    x_trunk = x_trunk.to(device)
    y = y.to(device)

    # DeepONet defines a custom `train(train_data, ...)` API and has two "last"
    # linear layers (branch + trunk). For a robust last-layer Laplace fit,
    # freeze all but those layers and fit Laplace on the trainable subset.
    model_for_laplace = model
    subset_for_laplace = subset_of_weights
    use_last_layer_freeze = subset_of_weights == "last_layer"
    base_state = None
    all_params = None

    if use_last_layer_freeze:
        # Work on a copy to avoid mutating the caller model's requires_grad flags.
        model_for_laplace = copy.deepcopy(model).to(device)
        base_state = {k: v.detach().clone() for k, v in model_for_laplace.state_dict().items()}
        all_params = list(model_for_laplace.parameters())
        model_for_laplace, _, prior_std_empirical = freezelayer(model_for_laplace, device)
        subset_for_laplace = "all"

        # For DeepONet outputs with thousands of points, laplace-torch's jacrev
        # path can OOM even for last-layer subsets. Use the existing sampled
        # Gauss-Newton diagonal approximation instead in this case.
        if hessian_structure == "diag":
            if isinstance(prior_precision, torch.Tensor):
                prior_precision_value = float(prior_precision.detach().cpu().item())
            else:
                prior_precision_value = float(prior_precision)
            if prior_precision_value <= 0.0:
                raise ValueError("prior_precision must be > 0 for Laplace approximation.")

            # Match the original DeepONet LA setup: use empirical std of current
            # trainable (last-layer) parameters as base prior scale and interpret
            # `prior_precision` as a multiplicative strength factor.
            prior_std_empirical = max(float(prior_std_empirical), 1e-12)
            prior_std = prior_std_empirical / np.sqrt(prior_precision_value)

            # Moderately loosened memory limits for better curvature quality.
            manual_batch_size = max(1, min(batch_size, manual_max_batch_size))
            sample_points_per_batch = min(
                int(x_trunk.shape[0]),
                int(max(1, manual_sample_points_per_batch)),
            )

            print("Using manual diagonal Laplace for DeepONet last-layer fit (memory-safe path)...")
            print(f"  manual_batch_size={manual_batch_size}, sample_points_per_batch={sample_points_per_batch}")
            print(
                f"  empirical_prior_std={prior_std_empirical:.3e}, "
                f"effective_prior_std={prior_std:.3e}, prior_precision_multiplier={prior_precision_value:.3e}"
            )

            # Keep deterministic behavior (e.g., disable dropout) while
            # computing curvature terms.
            nn.Module.train(model_for_laplace, False)
            H_diag = compute_diagonal_hessian(
                model_for_laplace,
                x_branch,
                x_trunk,
                y,
                noise_std=noise_std,
                prior_std=prior_std,
                device=device,
                batch_size=manual_batch_size,
                sample_points_per_batch=sample_points_per_batch,
            )

            trainable_params = [p for p in model_for_laplace.parameters() if p.requires_grad]
            theta_map = parameters_to_vector(trainable_params).detach()
            H_flat = H_diag.flatten()
            if H_flat.numel() != theta_map.numel():
                raise ValueError(
                    f"Shape mismatch in manual Laplace: Hessian={H_flat.numel()}, params={theta_map.numel()}."
                )

            posterior_std_vec = torch.rsqrt(torch.clamp(H_flat, min=1e-12))
            la = _ManualDiagLaplace(theta_map, posterior_std_vec)

            # Attach context for reconstructing full vectors from sampled trainable subset.
            la._uq_last_layer_only = True
            la._uq_model_ref = model_for_laplace
            la._uq_base_state = base_state
            la._uq_all_params = all_params
            return la

    wrapped_model = _DeepONetBranchWrapper(model_for_laplace, x_trunk)
    dataset = TensorDataset(x_branch, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Fitting Laplace posterior with laplace-torch...")
    if isinstance(prior_precision, torch.Tensor):
        prior_precision_print = float(prior_precision.detach().cpu().item())
    else:
        prior_precision_print = float(prior_precision)
    print(f"  subset_of_weights={subset_of_weights}, effective_subset={subset_for_laplace}, hessian_structure={hessian_structure}")
    print(f"  batch_size={batch_size}, sigma_noise={noise_std:.3e}, prior_precision={prior_precision_print:.3e}")

    la = Laplace(
        wrapped_model,
        likelihood="regression",
        subset_of_weights=subset_for_laplace,
        hessian_structure=hessian_structure,
        sigma_noise=noise_std,
        prior_precision=prior_precision,
    )
    la.fit(loader)

    # Attach context for reconstructing full vectors when fitting only last layers.
    la._uq_last_layer_only = use_last_layer_freeze
    if use_last_layer_freeze:
        la._uq_model_ref = model_for_laplace
        la._uq_base_state = base_state
        la._uq_all_params = all_params

    return la


def sample_laplace_torch(la, num_samples=50):
    """
    Sample full-parameter vectors from a fitted Laplace posterior.

    Returns:
        Tensor with shape [num_samples, n_full_params], on CPU.
    """
    samples = la.sample(num_samples)

    # If Laplace was fit on DeepONet last layers only, reconstruct full vectors
    # so downstream code can load complete model parameters.
    if getattr(la, "_uq_last_layer_only", False):
        model_ref = la._uq_model_ref
        base_state = la._uq_base_state
        all_params = la._uq_all_params
        device = next(model_ref.parameters()).device
        full_samples = []
        for s in samples:
            full_samples.append(build_full_vector(model_ref, base_state, all_params, s.to(device)))
        return torch.stack(full_samples, dim=0)

    return samples.detach().cpu()


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


def uqevaluation(
    eval_indices,
    test_data,
    model,
    method,
    hmc_samples=None,
    la_samples=None,
    sgld_samples=None,
    swag_samples=None,
    model_ensemble=None,
    noise_std=0.2,
    epoch_mcd=100,
    max_posterior_samples=None,
    return_preds=False,
    flatten_output=False,
):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_branch_eval = test_data['X_train'][eval_indices]
    x_trunk_eval = test_data['X_trunk']
    y_eval = test_data['Y_train'][eval_indices]

    x_b = torch.from_numpy(x_branch_eval).float().to(device)
    x_t = torch.from_numpy(x_trunk_eval).float().to(device)
    preds_eval_list = []

    if method == 'hmc':
        if hmc_samples is None:
            raise ValueError("hmc_samples must be provided for method='hmc'.")
        draws = _subsample_draws(hmc_samples, max_posterior_samples)
        with torch.no_grad():
            for s in draws:
                unpack_params(model, s.to(device))
                pred = model.predict(x_b, x_t)
                preds_eval_list.append(pred.cpu().numpy())
    elif method == 'sgld':
        if sgld_samples is None:
            raise ValueError("sgld_samples must be provided for method='sgld'.")
        draws = _subsample_draws(sgld_samples, max_posterior_samples)
        with torch.no_grad():
            for s in draws:
                unpack_params(model, s.to(device))
                pred = model.predict(x_b, x_t)
                preds_eval_list.append(pred.cpu().numpy())
    elif method == 'mcd':
        with torch.no_grad():
            for _ in range(epoch_mcd):
                pred = model.forward(x_b, x_t)
                preds_eval_list.append(pred.cpu().numpy())
    elif method == 'la':
        if la_samples is None:
            raise ValueError("la_samples must be provided for method='la'.")
        draws = _subsample_draws(la_samples, max_posterior_samples)
        with torch.no_grad():
            for s in draws:
                unpack_params(model, s.to(device))
                pred = model.predict(x_b, x_t)
                preds_eval_list.append(pred.cpu().numpy())
    elif method == 'swag':
        if swag_samples is None:
            raise ValueError("swag_samples must be provided for method='swag'.")
        draws = _subsample_draws(swag_samples, max_posterior_samples)
        with torch.no_grad():
            for s in draws:
                unpack_params(model, s.to(device))
                pred = model.predict(x_b, x_t)
                preds_eval_list.append(pred.cpu().numpy())
    elif method == 'de':
        if model_ensemble is None:
            raise ValueError("model_ensemble must be provided for method='de'.")
        with torch.no_grad():
            for paths in model_ensemble:  # model is a list of paths for models.
                m = torch.load(paths, weights_only=False).to(device)
                pred = m.predict(x_b, x_t)
                preds_eval_list.append(pred.cpu().numpy())
    else:
        raise ValueError(f"Unknown UQ method: {method}")

    preds_eval = np.stack(preds_eval_list)
    if flatten_output:
        preds_eval = preds_eval.reshape(preds_eval.shape[0], preds_eval.shape[1], -1)
    if return_preds:
        return preds_eval, y_eval

    return uq_evaluation.compute_metric(preds_eval, noise_std, y_eval)

def baseline(eval_indices, test_data, model):
    
    x_branch_eval = test_data['X_train'][eval_indices]
    x_trunk_eval = test_data['X_trunk']
    y_eval = test_data['Y_train'][eval_indices]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        x_b = torch.from_numpy(x_branch_eval).float().to(device)
        x_t = torch.from_numpy(x_trunk_eval).float().to(device)
        pred = model.predict(x_b, x_t)
        pred_np = pred.cpu().numpy()
    errors = y_eval - pred_np
    squared_errors = errors ** 2
    rmse = np.sqrt(np.mean(squared_errors))
    return rmse
