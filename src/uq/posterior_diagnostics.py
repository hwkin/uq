import numpy as np
import torch
import matplotlib.pyplot as plt


def _to_numpy_draws(draws):
    if torch.is_tensor(draws):
        draws = draws.detach().cpu().numpy()
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim == 1:
        draws = draws.reshape(-1, 1)
    if draws.ndim > 2:
        draws = draws.reshape(draws.shape[0], -1)
    if draws.ndim != 2:
        raise ValueError(f"Expected draws with 2 dimensions after flattening, got shape {draws.shape}.")
    return draws


def _split_rhat_1d(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if n < 8:
        return np.nan
    if n % 2 == 1:
        x = x[1:]
        n -= 1

    half = n // 2
    chains = np.stack([x[:half], x[half:]], axis=0)
    chain_means = np.mean(chains, axis=1)
    B = half * np.var(chain_means, ddof=1)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    if not np.isfinite(W) or W <= 1e-14:
        return np.nan

    var_hat = ((half - 1.0) / half) * W + B / half
    if var_hat <= 0:
        return np.nan
    return float(np.sqrt(var_hat / W))


def _ess_1d(x, max_lag=None):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if n < 4:
        return np.nan

    x = x - np.mean(x)
    var = np.var(x)
    if not np.isfinite(var) or var <= 1e-14:
        return float(n)

    if max_lag is None:
        max_lag = min(n - 1, max(10, n // 2))
    else:
        max_lag = min(int(max_lag), n - 1)
    if max_lag <= 1:
        return float(n)

    acov = np.correlate(x, x, mode="full")[n - 1:n + max_lag] / n
    rho = acov / acov[0]

    # Geyer initial positive sequence on paired autocorrelations.
    rho_sum = 0.0
    k = 1
    while (2 * k) < rho.shape[0]:
        pair = rho[2 * k - 1] + rho[2 * k]
        if not np.isfinite(pair) or pair <= 0:
            break
        rho_sum += pair
        k += 1

    tau = 1.0 + 2.0 * rho_sum
    if not np.isfinite(tau) or tau <= 0:
        return np.nan
    ess = n / tau
    return float(np.clip(ess, 1.0, float(n)))


def _diagnostic_series(draws, projection_count=6, random_seed=0):
    rng = np.random.default_rng(random_seed)
    n_samples, n_params = draws.shape
    series = []

    # Global summaries
    series.append(draws.mean(axis=1))
    series.append(np.linalg.norm(draws, axis=1))

    # A few direct parameter traces
    direct_dims = min(4, n_params)
    for d in range(direct_dims):
        series.append(draws[:, d])

    # Random projections on a capped parameter subset for speed.
    cap = min(n_params, 256)
    if n_params > cap:
        idx = rng.choice(n_params, size=cap, replace=False)
        draws_sub = draws[:, idx]
    else:
        draws_sub = draws
        cap = n_params

    n_proj = min(int(projection_count), max(0, cap))
    for _ in range(n_proj):
        v = rng.normal(size=cap)
        v_norm = np.linalg.norm(v)
        if v_norm <= 0:
            continue
        v = v / v_norm
        series.append(draws_sub @ v)

    # Remove any non-finite traces.
    clean = []
    for s in series:
        s = np.asarray(s, dtype=np.float64).reshape(n_samples)
        if np.all(np.isfinite(s)):
            clean.append(s)
    return clean


def summarize_draws_diagnostics(
    draws,
    method_name="hmc",
    projection_count=6,
    random_seed=0,
    acceptance_tol=1e-12,
):
    draws_np = _to_numpy_draws(draws)
    n_samples, n_params = draws_np.shape

    traces = _diagnostic_series(
        draws_np,
        projection_count=projection_count,
        random_seed=random_seed,
    )

    ess_vals = np.array([_ess_1d(t) for t in traces], dtype=np.float64)
    rhat_vals = np.array([_split_rhat_1d(t) for t in traces], dtype=np.float64)
    ess_vals = ess_vals[np.isfinite(ess_vals)]
    rhat_vals = rhat_vals[np.isfinite(rhat_vals)]

    diag = {
        "n_samples": int(n_samples),
        "n_params": int(n_params),
        "ess_mean": float(np.mean(ess_vals)) if ess_vals.size else np.nan,
        "ess_min": float(np.min(ess_vals)) if ess_vals.size else np.nan,
        "rhat_mean": float(np.mean(rhat_vals)) if rhat_vals.size else np.nan,
        "rhat_max": float(np.max(rhat_vals)) if rhat_vals.size else np.nan,
        "acceptance_rate": np.nan,
        "acceptance_trace": None,
    }

    method_key = str(method_name).strip().lower()
    if method_key in {"hmc", "sgld"} and n_samples > 1:
        delta = np.linalg.norm(np.diff(draws_np, axis=0), axis=1)
        accepted = (delta > acceptance_tol).astype(np.float64)
        diag["acceptance_rate"] = float(np.mean(accepted))
        # Running acceptance rate trace (same length as draws).
        running = np.concatenate(
            [[accepted[0]], np.cumsum(accepted) / np.arange(1, accepted.shape[0] + 1)]
        )
        diag["acceptance_trace"] = running

    return diag


def format_diagnostics(method_label, diag):
    acc_rate = diag.get("acceptance_rate", np.nan)
    if np.isfinite(acc_rate):
        acc_str = f"{acc_rate:.3f}"
    else:
        acc_str = "n/a"
    return (
        f"[{method_label}] n={diag.get('n_samples')}, p={diag.get('n_params')}, "
        f"ESS(mean/min)=({diag.get('ess_mean', np.nan):.1f}/{diag.get('ess_min', np.nan):.1f}), "
        f"Rhat(mean/max)=({diag.get('rhat_mean', np.nan):.3f}/{diag.get('rhat_max', np.nan):.3f}), "
        f"acc={acc_str}"
    )


def plot_acceptance_trace(trace, title="Acceptance Trace", window=50):
    if trace is None:
        return
    trace = np.asarray(trace, dtype=np.float64).reshape(-1)
    if trace.size == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(trace, lw=1.5, label="Running acceptance")

    w = max(1, int(window))
    if trace.size >= w:
        kernel = np.ones(w, dtype=np.float64) / w
        smooth = np.convolve(trace, kernel, mode="valid")
        x = np.arange(w - 1, trace.size)
        ax.plot(x, smooth, lw=1.5, label=f"Window-{w} mean")

    ax.set_title(title)
    ax.set_xlabel("Draw index")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
