import numpy as np
import matplotlib.pyplot as plt

def compute_metric(preds_eval, noise_std, y_eval):
    # Compute uncertainties
    mean_pred_eval = preds_eval.mean(axis=0)
    epistemic_var_eval = preds_eval.var(axis=0)
    epistemic_std_eval = np.sqrt(epistemic_var_eval)
    aleatoric_var_eval = noise_std ** 2
    total_var_eval = epistemic_var_eval + aleatoric_var_eval
    total_std_eval = np.sqrt(total_var_eval)
    sample_std = np.max(epistemic_std_eval, axis=1)

    # PREDICTION ERROR
    errors = y_eval - mean_pred_eval
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

    # Return epistemic standard deviation (useful for OOD detection) and summary metrics
    return sample_std, np.array([rmse, coverage_1sigma, coverage_2sigma, coverage_3sigma, mpiw, nll, np.mean(total_std_eval)])

def comparison_uq(result_lst, method_lst):
    results = {}
    for i, method in enumerate(method_lst):
        results[method] = result_lst[i]
    comparison_data = {'Metric': ['RMSE', 'Coverage 1σ (%)','Coverage 2σ (%)','Coverage 3σ (%)','MPIW', 'NLL'],\
        'Ideal': [ 'Lower', '68.3', '95.4', '99.7', 'Lower', 'Lower']}
    for method, res in results.items():
        rmse = res[0]
        cov1 = res[1] * 100
        cov2 = res[2] * 100
        cov3 = res[3] * 100
        mpiw = res[4]
        nll = res[5]
        comparison_data[method] = [f"{rmse:.4f}", f"{cov1:.2f}", f"{cov2:.2f}", f"{cov3:.2f}", f"{mpiw:.4f}", f"{nll:.4f}"]

    # Print comparison table
    method_names = list(results.keys())
    metric_col_width = 25
    method_col_width = 12
    ideal_col_width = 10

    widths = [metric_col_width] + [method_col_width] * len(method_names) + [ideal_col_width]
    total_width = sum(widths) + (len(widths) - 1)
    row_fmt = (
        f"{{:<{metric_col_width}}} "
        + " ".join([f"{{:>{method_col_width}}}"] * len(method_names))
        + f" {{:>{ideal_col_width}}}"
    )

    print("\n" + row_fmt.format('Metric', *method_names, 'Ideal'))
    print("-" * total_width)
    for i in range(len(comparison_data['Metric'])):
        print(row_fmt.format(
            comparison_data['Metric'][i],
            *(comparison_data[method][i] for method in method_names),
            comparison_data['Ideal'][i]
            ))

def run_regression_shift(
    method,
    levels,
    results_id,
    results_shifting,
    baseline_rmse_id,
    baseline_rmse_copy,
    id_index=0,
):
    stats = {m: {'rmse': [], 'mpiw':[],'nll': [], 'unc': [], 'cov': []} for m in method}

    line_styles = {
        'HMC': {'color': 'C0', 'linestyle': '-', 'linewidth': 2.5},
        'MC Dropout': {'color': 'C1', 'linestyle': '--', 'linewidth': 2.25},
        'laplace Approximation': {'color': 'C2', 'linestyle': '-.', 'linewidth': 2.25},
        'Deep Ensemble': {'color': 'C3', 'linestyle': ':', 'linewidth': 2.5},
        'SGLD': {'color': 'C4', 'linestyle': '-', 'linewidth': 2.5},
    }

    for met, result in zip(method, results_shifting):
        for i, lvl in enumerate(levels):
            stats[met]['rmse'].append(result[i][0])
            stats[met]['cov'].append(result[i][2])
            stats[met]['mpiw'].append(result[i][4])
            stats[met]['nll'].append(result[i][5])
            stats[met]['unc'].append(result[i][6])

    subplot_width = 4.0
    legend_width = 2.5 + 0.15 * max(0, len(method) - 4)
    fig_width = subplot_width * 5 + legend_width
    fig = plt.figure(figsize=(fig_width, 4))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, legend_width / subplot_width], wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    legend_ax = fig.add_subplot(gs[0, 5])
    legend_ax.axis('off')

    metrics = ['rmse', 'cov', 'mpiw', 'nll', 'unc']
    titles = ['RMSE (Error) (↓)', '95% Coverage (Target: 0.95)' , 'MPIW (↓)', 'NLL (↓)', 'Uncertainty (Avg Std) (↑)']
    metric_to_idx = {'rmse': 0, 'cov': 2, 'mpiw': 4, 'nll': 5, 'unc': 6}
    x_vals = np.arange(len(levels))

    if id_index is not None and not (0 <= id_index < len(levels)):
        raise ValueError("id_index is out of range for levels.")

    default_colors = plt.rcParams.get('axes.prop_cycle', None)
    default_colors = default_colors.by_key().get('color', []) if default_colors else []
    default_linestyles = ['-', '--', '-.', ':']
    for met_i, met in enumerate(method):
        if met not in line_styles:
            color = default_colors[met_i % len(default_colors)] if default_colors else None
            linestyle = default_linestyles[(met_i // max(1, len(default_colors))) % len(default_linestyles)]
            line_styles[met] = {'color': color, 'linestyle': linestyle, 'linewidth': 2}

    method_handles = {}
    baseline_handle = None
    ideal_handle = None

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for met, data in stats.items():
            style = line_styles.get(met, {})
            line = ax.plot(
                x_vals,
                data[metric],
                marker='o',
                label=met,
                color=style.get('color'),
                linestyle=style.get('linestyle', '-'),
                linewidth=style.get('linewidth', 2)
            )[0]
            if metric == 'rmse' and met not in method_handles:
                method_handles[met] = line

        if metric == 'rmse' and baseline_rmse_copy is not None:
            if np.isscalar(baseline_rmse_copy):
                baseline_vals = [baseline_rmse_copy] * len(levels)
            else:
                baseline_vals = list(baseline_rmse_copy)
                if len(baseline_vals) != len(levels):
                    raise ValueError("baseline_rmse_copy must be a scalar or have the same length as levels.")
            baseline_handle = ax.plot(
                x_vals,
                baseline_vals,
                marker='o',
                label='Baseline',
                color='black',
                linestyle=':',
                linewidth=2
            )[0]

        if id_index is not None and results_id is not None:
            idx = metric_to_idx[metric]
            for met_i, met in enumerate(method):
                if isinstance(results_id, dict):
                    res_id = results_id[met]
                else:
                    res_id = results_id[met_i]
                y_val = res_id[idx]
                style = line_styles.get(met, {})
                ax.scatter(
                    x_vals[id_index],
                    y_val,
                    s=90,
                    marker='o',
                    color=style.get('color'),
                    edgecolor='black',
                    linewidths=1.0,
                    zorder=5,
                    label=None
                )
            if metric == 'rmse' and baseline_rmse_id is not None:
                ax.scatter(
                    x_vals[id_index],
                    baseline_rmse_id,
                    s=100,
                    marker='o',
                    color='black',
                    edgecolor='white',
                    linewidths=1.0,
                    zorder=6,
                    label=None
                )

        # Draw target line for coverage
        if metric == 'cov':
            ideal_handle = ax.axhline(0.95, color='black', linestyle='--', label='Ideal')

        ax.set_title(titles[i])
        ax.set_xlabel('Shift Intensity')
        ax.set_xticks(x_vals)
        ax.set_xticklabels(levels)
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.grid(True, alpha=0.3)

    legend_handles = [method_handles[m] for m in method if m in method_handles]
    legend_labels = [m for m in method if m in method_handles]
    if baseline_handle is not None:
        legend_handles.append(baseline_handle)
        legend_labels.append('Baseline')
    if ideal_handle is not None:
        legend_handles.append(ideal_handle)
        legend_labels.append('Ideal')

    if legend_handles:
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc='center left',
            frameon=False,
            labelspacing=0.8,
            handlelength=2.0
        )

    fig.tight_layout()
    plt.show()
