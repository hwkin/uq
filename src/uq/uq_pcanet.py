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


# --- Log-posterior: Gaussian prior + Gaussian likelihood for PCANet ---
def make_log_prob_fn(model, X, y, noise_std=0.01, prior_std=1.0):
    """
    Create log probability function for HMC sampling.

    Args:
        model: PCANet model
        X: Input data, numpy array or tensor
        y: Target output, numpy array or tensor
        noise_std: Observation noise standard deviation
        prior_std: Prior standard deviation for weights
    """
    device = next(model.parameters()).device

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    X = X.to(device)
    y = y.to(device)

    def log_prob(flat_params):
        unpack_params(model, flat_params)
        pred = model.forward(X)
        resid = (y - pred).reshape(y.shape[0], -1)

        ll = -0.5 * (resid.pow(2).sum() / (noise_std**2))
        lp = -0.5 * (flat_params.pow(2).sum() / (prior_std**2))
        return ll + lp

    return log_prob


def make_minibatch_log_prob_fn(model, X, y, batch_size=100, noise_std=0.01, prior_std=1.0):
    """
    Create stochastic log probability function for SGLD with mini-batching.
    """
    device = next(model.parameters()).device

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    X = X.to(device)
    y = y.to(device)

    N = X.shape[0]

    def log_prob(flat_params):
        unpack_params(model, flat_params)

        idx = torch.randperm(N, device=device)[: min(batch_size, N)]
        X_batch = X[idx]
        y_batch = y[idx]
        batch_n = X_batch.shape[0]

        pred = model.forward(X_batch)
        resid = (y_batch - pred).reshape(batch_n, -1)

        sse = resid.pow(2).sum()
        ll = -0.5 * (N / batch_n) * (sse / (noise_std**2))
        lp = -0.5 * (flat_params.pow(2).sum() / (prior_std**2))
        return ll + lp

    return log_prob


def freezelayer(model, device):
    # Freeze all parameters except the last MLP layer.
    for param in model.parameters():
        param.requires_grad = False

    if not hasattr(model, "net") or not hasattr(model.net, "layers"):
        raise AttributeError("Expected model.net.layers to exist for last-layer freezing.")

    for param in model.net.layers[-1].parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        raise RuntimeError("No trainable parameters after freezelayer().")

    print(f"Trainable params: {n_trainable}")
    flat0 = pack_params(model).to(device)
    print(f"Initial parameter vector shape (trainable only): {flat0.shape}")

    if flat0.numel() > 1:
        param_std_last_init = torch.std(flat0)
        prior_std = float(param_std_last_init.item())
    else:
        prior_std = 1.0
    prior_std = max(prior_std, 1e-12)
    print(f"Initial last-layer param std (scalar): {prior_std:.3e}")

    return model, flat0, prior_std


def build_full_vector(model, base_state, all_params, trainable_flat):
    offset = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(base_state[name])
        for p in all_params:
            if p.requires_grad:
                numel = p.numel()
                p.copy_(trainable_flat[offset : offset + numel].view_as(p))
                offset += numel

    if offset != trainable_flat.numel():
        raise ValueError(
            f"Mismatch while rebuilding full vector: consumed {offset}, provided {trainable_flat.numel()}."
        )

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
def hmc_nuts(
    log_prob_fn,
    initial,
    initial_step_size=1e-4,
    leapfrog_steps=20,
    num_samples=1000,
    burn_in=1000,
    random_seed=42,
):
    """
    Hamiltonian Monte Carlo sampler using Hamiltorch.
    """
    print("Starting HMC with hamiltorch...")
    print(f"  Samples: {num_samples} + Burn-in: {burn_in}")
    hamiltorch.set_random_seed(random_seed)

    params_hmc = hamiltorch.sample(
        log_prob_func=log_prob_fn,
        params_init=initial,
        num_samples=num_samples,
        burn=burn_in,
        step_size=initial_step_size,
        num_steps_per_sample=leapfrog_steps,
        sampler=hamiltorch.Sampler.HMC_NUTS,
    )

    samples = torch.stack(params_hmc)
    return samples


def fit_hmc_torch(
    model,
    X,
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
    Run HMC sampling for PCANet and return full parameter vectors.

    Args:
        model: Trained PCANet model.
        X: Inputs used for log-likelihood.
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
        X,
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

    if reconstruct_full:
        full_samples = []
        for s in sampled:
            full_samples.append(build_full_vector(model_for_hmc, base_state, all_params, s.detach()))
        return torch.stack(full_samples, dim=0)

    return sampled.detach().cpu()


def sgld(
    log_prob_fn,
    initial,
    step_size=1e-5,
    num_samples=500,
    burn_in=100,
    step_decay=0.9999,
    min_step_size=1e-7,
    grad_clip=10.0,
    random_seed=42,
):
    """
    Stochastic Gradient Langevin Dynamics (SGLD) sampler.
    """
    torch.manual_seed(random_seed)
    samples = []
    current = initial.clone().detach().requires_grad_(True)

    total_iterations = num_samples + burn_in
    eps = step_size

    print("Starting SGLD sampling...")
    print(f"  Burn-in: {burn_in}, Samples: {num_samples}")
    print(f"  Initial step size: {step_size:.2e}")

    for i in range(total_iterations):
        lp = log_prob_fn(current)
        if not torch.isfinite(lp):
            print(f"Warning: non-finite log_prob at iter {i + 1}; stopping early.")
            break

        grad = torch.autograd.grad(lp, current, create_graph=False)[0]
        if not torch.isfinite(grad).all():
            print(f"Warning: non-finite gradient at iter {i + 1}; stopping early.")
            break

        if grad_clip is not None:
            grad_norm = grad.norm()
            if grad_norm > grad_clip:
                grad = grad * (grad_clip / grad_norm)

        noise = torch.randn_like(current) * np.sqrt(eps)
        current = (current + 0.5 * eps * grad + noise).detach().requires_grad_(True)

        eps = max(eps * step_decay, min_step_size)

        if i >= burn_in:
            samples.append(current.detach().clone())

        if (i + 1) % 100 == 0:
            phase = "burn-in" if i < burn_in else "sampling"
            print(f"Iter {i + 1:4d}/{total_iterations}: step_size = {eps:.2e}, phase = {phase}")

    if len(samples) == 0:
        raise RuntimeError("SGLD produced zero samples. Check step size and numerical stability.")

    samples = torch.stack(samples)
    print(f"SGLD completed. Collected {len(samples)} samples.")
    return samples, eps


# --- SWA-Gaussian (SWAG) ---
def _collect_swag_snapshot(model, swag_state, max_rank):
    """
    Update SWAG running statistics with the current full-parameter snapshot.
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
    Draw parameter samples from the SWAG Gaussian posterior approximation.

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
class _PCANetWrapper(nn.Module):
    """
    Adapter so laplace-torch can call a one-input PCANet while avoiding
    PCANet.train(...) overload conflicts.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward(x)

    @staticmethod
    def _set_training_flag(module, mode):
        module.training = mode
        for child in module.children():
            _PCANetWrapper._set_training_flag(child, mode)

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
    batch_size=20,
    noise_std=0.2,
    prior_precision=1.0,
    subset_of_weights="last_layer",
    hessian_structure="diag",
    random_seed=42,
):
    """
    Fit a Laplace posterior using laplace-torch.

    Args:
        model: Trained PCANet model.
        X: Inputs (numpy array or tensor), shape [N, ...].
        y: Targets (numpy array or tensor), shape [N, ...].
        batch_size: Batch size for Hessian accumulation.
        noise_std: Observation noise std used by regression Laplace likelihood.
        prior_precision: Prior precision (inverse variance) for weights.
        subset_of_weights: Which parameter subset to fit.
            Use 'last_layer' or 'full_layer'/'all'.
        hessian_structure: laplace-torch Hessian structure, e.g., 'diag' or 'kron'.
        random_seed: Random seed.

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

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    else:
        X = X.float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    else:
        y = y.float()

    X = X.to(device)
    y = y.to(device)

    model_for_laplace = model
    subset_for_laplace = subset_of_weights
    use_last_layer_freeze = subset_of_weights == "last_layer"
    base_state = None
    all_params = None

    if use_last_layer_freeze:
        model_for_laplace = copy.deepcopy(model).to(device)
        base_state = {k: v.detach().clone() for k, v in model_for_laplace.state_dict().items()}
        all_params = list(model_for_laplace.parameters())
        model_for_laplace, _, _ = freezelayer(model_for_laplace, device)
        subset_for_laplace = "all"

    wrapped_model = _PCANetWrapper(model_for_laplace)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Fitting Laplace posterior with laplace-torch...")
    if isinstance(prior_precision, torch.Tensor):
        prior_precision_print = float(prior_precision.detach().cpu().item())
    else:
        prior_precision_print = float(prior_precision)
    print(
        f"  subset_of_weights={subset_of_weights}, effective_subset={subset_for_laplace}, "
        f"hessian_structure={hessian_structure}"
    )
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
# Compute diagonal Gauss-Newton Hessian approximation
# ============================================================
def compute_diagonal_hessian(
    model,
    X,
    y,
    noise_std,
    prior_std,
    device,
    batch_size=20,
    sample_outputs_per_batch=50,
):
    """
    Compute diagonal approximation of the Hessian using a Gauss-Newton method.
    """
    if sample_outputs_per_batch <= 0:
        raise ValueError("sample_outputs_per_batch must be positive.")

    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    X_tensor = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X.clone().float()
    y_tensor = torch.from_numpy(y).float() if isinstance(y, np.ndarray) else y.clone().float()

    n_samples = X_tensor.shape[0]
    n_outputs = y_tensor.reshape(y_tensor.shape[0], -1).shape[1]

    H_diag = torch.ones(n_params, device=device) / (prior_std**2)

    sample_count = min(sample_outputs_per_batch, n_outputs)
    scale_factor = n_outputs / sample_count
    noise_var_inv = 1.0 / (noise_std**2)

    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        X_batch = X_tensor[i:batch_end].to(device)

        sample_indices = np.random.choice(n_outputs, sample_count, replace=False)

        pred = model.forward(X_batch).reshape(X_batch.shape[0], -1)
        batch_size_actual = pred.shape[0]
        for j in range(batch_size_actual):
            for idx, k in enumerate(sample_indices):
                model.zero_grad()
                is_last = (j == batch_size_actual - 1) and (idx == len(sample_indices) - 1)
                pred[j, k].backward(retain_graph=not is_last)

                grad_sq_sum = torch.zeros(n_params, device=device)
                offset = 0
                for p in params:
                    numel = p.numel()
                    if p.grad is not None:
                        grad_sq_sum[offset : offset + numel] = p.grad.view(-1).pow(2)
                    offset += numel

                H_diag += grad_sq_sum * noise_var_inv * scale_factor

        del pred, X_batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if (i + batch_size) % 100 == 0 or batch_end == n_samples:
            print(f"  Processed {batch_end}/{n_samples} samples")

    return H_diag


def inject_dropout(model, target_layer_type=nn.Linear, dropout_rate=0.1):
    """
    Recursively add a Dropout layer after every occurrence of `target_layer_type`.
    """
    for name, child in model.named_children():
        if isinstance(child, target_layer_type):
            new_layer = nn.Sequential(
                child,
                nn.Dropout(dropout_rate),
            )
            setattr(model, name, new_layer)
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
    """
    Evaluate predictive uncertainty metrics for PCANet.

    `eval_indices` can be an index array (preferred) or an int (legacy API).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_total = len(test_data["X_train"])
    eval_indices = _resolve_eval_indices(eval_indices, n_total)

    x_eval = _take_rows(test_data["X_train"], eval_indices)
    y_eval = _take_rows(test_data["Y_train"], eval_indices)

    x_tensor = _to_tensor_float(x_eval, device)
    y_eval_np = _to_numpy(y_eval)
    preds_eval_list = []

    if method == "hmc":
        if hmc_samples is None:
            raise ValueError("hmc_samples must be provided for method='hmc'.")
        draws = _subsample_draws(hmc_samples, max_posterior_samples)
        with torch.no_grad():
            for s in draws:
                unpack_params(model, s.to(device))
                pred = model.predict(x_tensor)
                preds_eval_list.append(pred.detach().cpu().numpy())
    elif method == "sgld":
        if sgld_samples is None:
            raise ValueError("sgld_samples must be provided for method='sgld'.")
        draws = _subsample_draws(sgld_samples, max_posterior_samples)
        with torch.no_grad():
            for s in draws:
                unpack_params(model, s.to(device))
                pred = model.predict(x_tensor)
                preds_eval_list.append(pred.detach().cpu().numpy())
    elif method == "mcd":
        with torch.no_grad():
            for _ in range(epoch_mcd):
                pred = model.forward(x_tensor)
                preds_eval_list.append(pred.detach().cpu().numpy())
    elif method == "la":
        if la_samples is None:
            raise ValueError("la_samples must be provided for method='la'.")
        draws = _subsample_draws(la_samples, max_posterior_samples)
        with torch.no_grad():
            for s in draws:
                unpack_params(model, s.to(device))
                pred = model.predict(x_tensor)
                preds_eval_list.append(pred.detach().cpu().numpy())
    elif method == "swag":
        if swag_samples is None:
            raise ValueError("swag_samples must be provided for method='swag'.")
        draws = _subsample_draws(swag_samples, max_posterior_samples)
        with torch.no_grad():
            for s in draws:
                unpack_params(model, s.to(device))
                pred = model.predict(x_tensor)
                preds_eval_list.append(pred.detach().cpu().numpy())
    elif method == "de":
        if model_ensemble is None:
            raise ValueError("model_ensemble must be provided for method='de'.")
        draws = _subsample_draws(model_ensemble, max_posterior_samples)
        with torch.no_grad():
            for path in draws:
                m = torch.load(path, weights_only=False).to(device)
                pred = m.predict(x_tensor)
                preds_eval_list.append(pred.detach().cpu().numpy())
    else:
        raise ValueError(f"Unknown UQ method: {method}")

    preds_eval = np.stack(preds_eval_list)
    if flatten_output:
        preds_eval = preds_eval.reshape(preds_eval.shape[0], preds_eval.shape[1], -1)
    if return_preds:
        return preds_eval, y_eval_np

    return uq_evaluation.compute_metric(preds_eval, noise_std, y_eval_np)


def baseline(eval_indices, test_data, model):
    """
    Baseline deterministic RMSE using MAP model predictions.

    `eval_indices` can be an index array (preferred) or an int (legacy API).
    """
    n_total = len(test_data["X_train"])
    eval_indices = _resolve_eval_indices(eval_indices, n_total)

    x_eval = _take_rows(test_data["X_train"], eval_indices)
    y_eval = _take_rows(test_data["Y_train"], eval_indices)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_tensor = _to_tensor_float(x_eval, device)

    with torch.no_grad():
        pred = model.predict(x_tensor)
        pred_np = pred.detach().cpu().numpy()

    y_eval_np = _to_numpy(y_eval)
    errors = y_eval_np - pred_np
    rmse = np.sqrt(np.mean(errors**2))
    return float(rmse)
