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
    sample_std = np.mean(epistemic_std_eval, axis=1)

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
    print("\n{:<25} {:>12} {:>12} {:>12} {:>10}".format('Metric', *results.keys(), 'Ideal'))
    print("-" * 85)
    for i in range(len(comparison_data['Metric'])):
        print("{:<25} {:>12} {:>12} {:>12} {:>10}".format(
            comparison_data['Metric'][i],
            *(comparison_data[method][i] for method in results.keys()),
            comparison_data['Ideal'][i]
            ))

def run_regression_shift(method, levels, results):
    stats = {m: {'rmse': [], 'mpiw':[],'nll': [], 'unc': [], 'cov': []} for m in method}

    line_styles = {
        'HMC': {'color': 'C0', 'linestyle': '-', 'linewidth': 2.5},
        'MC Dropout': {'color': 'C1', 'linestyle': '--', 'linewidth': 2.25},
        'laplace Approximation': {'color': 'C2', 'linestyle': '-.', 'linewidth': 2.25},
        'Deep Ensemble': {'color': 'C3', 'linestyle': ':', 'linewidth': 2.5},
        'SGLD': {'color': 'C4', 'linestyle': '-', 'linewidth': 2.5},
    }

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
            style = line_styles.get(met, {})
            ax.plot(
                levels,
                data[metric],
                marker='o',
                label=met,
                color=style.get('color'),
                linestyle=style.get('linestyle', '-'),
                linewidth=style.get('linewidth', 2)
            )
            
        # Draw target line for coverage
        if metric == 'cov':
            ax.axhline(0.95, color='black', linestyle='--', label='Ideal')
            
        ax.set_title(titles[i])
        ax.set_xlabel('Shift Intensity')
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()
        
    plt.tight_layout()
    plt.show()