import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def _setup_paths(repo_root: str):
    sys.path.extend(
        [
            os.path.join(repo_root, "src", "data"),
            os.path.join(repo_root, "src", "uq"),
            os.path.join(repo_root, "src", "nn", "fno"),
        ]
    )


def _load_assets(data_folder, num_train, num_test, num_y_components, coarsen_grid_factor):
    from dataMethods import DataProcessorFNO

    data = DataProcessorFNO(
        os.path.join(data_folder, "Poisson_FNO_samples_.npz"),
        num_train,
        num_test,
        num_y_components,
        coarsen_grid_factor,
    )
    train_data = {"X_train": data.X_train, "Y_train": data.Y_train}
    test_data = {"X_train": data.X_test, "Y_train": data.Y_test}
    return data, train_data, test_data


def _load_model(model_path, device: torch.device):
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.to(device)
    model.device = device
    return model


def _run_one(
    cfg,
    model_path,
    train_data,
    test_data,
    hmc_indices,
    eval_indices,
    seed,
    max_eval_draws,
    device,
):
    import uq_fno as uq
    import posterior_diagnostics as post_diag

    model = _load_model(model_path, device=device)

    x_hmc = train_data["X_train"][hmc_indices]
    y_hmc = train_data["Y_train"][hmc_indices]

    t0 = time.time()
    samples = uq.fit_hmc_torch(
        model,
        x_hmc,
        y_hmc,
        noise_std=cfg["noise_std"],
        subset_of_weights=cfg["subset_of_weights"],
        prior_std=cfg["prior_std"],
        initial_step_size=cfg["initial_step_size"],
        leapfrog_steps=cfg["leapfrog_steps"],
        num_samples=cfg["num_samples"],
        burn_in=cfg["burn_in"],
        random_seed=seed,
        batch_size=None,
        reduce_output_mean=cfg["reduce_output_mean"],
    )
    elapsed = time.time() - t0

    has_nan = bool(torch.isnan(samples).any())
    n_draws = int(samples.shape[0])

    diag = post_diag.summarize_draws_diagnostics(
        samples,
        method_name="hmc",
        random_seed=seed,
    )

    eval_model = _load_model(model_path, device=device)
    _, result = uq.uqevaluation(
        eval_indices,
        test_data,
        eval_model,
        "hmc",
        hmc_samples=samples,
        max_posterior_samples=min(max_eval_draws, n_draws),
    )
    # result = [rmse, cov1, cov2, cov3, mpiw, nll, mean_total_std]
    return {
        "name": cfg["name"],
        "subset_of_weights": cfg["subset_of_weights"],
        "noise_std": cfg["noise_std"],
        "prior_std": cfg["prior_std"],
        "initial_step_size": cfg["initial_step_size"],
        "leapfrog_steps": cfg["leapfrog_steps"],
        "reduce_output_mean": cfg["reduce_output_mean"],
        "num_samples": cfg["num_samples"],
        "burn_in": cfg["burn_in"],
        "elapsed_sec": elapsed,
        "n_draws": n_draws,
        "has_nan": has_nan,
        "acceptance_rate": float(diag.get("acceptance_rate", np.nan)),
        "ess_mean": float(diag.get("ess_mean", np.nan)),
        "ess_min": float(diag.get("ess_min", np.nan)),
        "rhat_mean": float(diag.get("rhat_mean", np.nan)),
        "rhat_max": float(diag.get("rhat_max", np.nan)),
        "rmse": float(result[0]),
        "cov1": float(result[1]),
        "cov2": float(result[2]),
        "cov3": float(result[3]),
        "mpiw": float(result[4]),
        "nll": float(result[5]),
        "mean_total_std": float(result[6]),
    }


def main():
    parser = argparse.ArgumentParser(description="Tune HMC settings for FNO posterior sampling.")
    parser.add_argument("--repo-root", default="/root/experiment")
    parser.add_argument("--data-folder", default="/root/autodl-tmp/data")
    parser.add_argument("--num-train", type=int, default=3500)
    parser.add_argument("--num-test", type=int, default=1000)
    parser.add_argument("--num-hmc-data", type=int, default=128)
    parser.add_argument("--num-eval-data", type=int, default=100)
    parser.add_argument("--num-y-components", type=int, default=1)
    parser.add_argument("--coarsen-grid-factor", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-eval-draws", type=int, default=40)
    parser.add_argument("--device", default="cpu", help="Device for HMC and evaluation (e.g., 'cpu', 'cuda:0').")
    parser.add_argument(
        "--out-json",
        default="/root/experiment/poisson/FNO/hmc_tuning_results.json",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False.")
    print(f"[Device] Using {device}")

    _setup_paths(args.repo_root)
    import uq_fno as uq

    data, train_data, test_data = _load_assets(
        args.data_folder,
        args.num_train,
        args.num_test,
        args.num_y_components,
        args.coarsen_grid_factor,
    )

    model_path = os.path.join(args.data_folder, "FNO", "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    hmc_indices = np.random.choice(args.num_train, args.num_hmc_data, replace=False)
    eval_indices = np.arange(min(args.num_eval_data, args.num_test), dtype=int)

    baseline_model = _load_model(model_path, device=device)
    baseline_rmse = uq.baseline(eval_indices, test_data, baseline_model)
    print(f"[Baseline MAP] eval_n={len(eval_indices)} rmse={baseline_rmse:.6f}")

    # Coarse grid focused on stability + predictive quality.
    configs = [
        {
            "name": "proj_n0.2_p1_rmTrue",
            "subset_of_weights": "projectors",
            "noise_std": 0.2,
            "prior_std": 1.0,
            "initial_step_size": 5e-6,
            "leapfrog_steps": 5,
            "reduce_output_mean": True,
            "num_samples": 120,
            "burn_in": 80,
        },
        {
            "name": "proj_n0.1_p1_rmTrue",
            "subset_of_weights": "projectors",
            "noise_std": 0.1,
            "prior_std": 1.0,
            "initial_step_size": 5e-6,
            "leapfrog_steps": 5,
            "reduce_output_mean": True,
            "num_samples": 120,
            "burn_in": 80,
        },
        {
            "name": "proj_n0.05_p1_rmTrue",
            "subset_of_weights": "projectors",
            "noise_std": 0.05,
            "prior_std": 1.0,
            "initial_step_size": 5e-6,
            "leapfrog_steps": 5,
            "reduce_output_mean": True,
            "num_samples": 120,
            "burn_in": 80,
        },
        {
            "name": "proj_n0.1_p0.5_rmTrue",
            "subset_of_weights": "projectors",
            "noise_std": 0.1,
            "prior_std": 0.5,
            "initial_step_size": 5e-6,
            "leapfrog_steps": 5,
            "reduce_output_mean": True,
            "num_samples": 120,
            "burn_in": 80,
        },
        {
            "name": "proj_n0.05_p0.5_rmTrue",
            "subset_of_weights": "projectors",
            "noise_std": 0.05,
            "prior_std": 0.5,
            "initial_step_size": 5e-6,
            "leapfrog_steps": 5,
            "reduce_output_mean": True,
            "num_samples": 120,
            "burn_in": 80,
        },
        {
            "name": "proj_n0.1_p1_rmFalse",
            "subset_of_weights": "projectors",
            "noise_std": 0.1,
            "prior_std": 1.0,
            "initial_step_size": 1e-7,
            "leapfrog_steps": 5,
            "reduce_output_mean": False,
            "num_samples": 120,
            "burn_in": 80,
        },
        {
            "name": "last_n0.1_p0.5_rmTrue",
            "subset_of_weights": "last_layer",
            "noise_std": 0.1,
            "prior_std": 0.5,
            "initial_step_size": 5e-6,
            "leapfrog_steps": 5,
            "reduce_output_mean": True,
            "num_samples": 120,
            "burn_in": 80,
        },
        {
            "name": "projbias_n0.2_p0.2_rmTrue",
            "subset_of_weights": "projector_biases",
            "noise_std": 0.2,
            "prior_std": 0.2,
            "initial_step_size": 1e-4,
            "leapfrog_steps": 5,
            "reduce_output_mean": True,
            "num_samples": 120,
            "burn_in": 80,
        },
        {
            "name": "outbias_n0.2_p0.2_rmTrue",
            "subset_of_weights": "output_bias",
            "noise_std": 0.2,
            "prior_std": 0.2,
            "initial_step_size": 1e-4,
            "leapfrog_steps": 5,
            "reduce_output_mean": True,
            "num_samples": 120,
            "burn_in": 80,
        },
        {
            "name": "realall_n0.1_p0.5_rmTrue",
            "subset_of_weights": "real_all",
            "noise_std": 0.1,
            "prior_std": 0.5,
            "initial_step_size": 5e-7,
            "leapfrog_steps": 5,
            "reduce_output_mean": True,
            "num_samples": 120,
            "burn_in": 80,
        },
    ]

    rows = []
    for i, cfg in enumerate(configs, start=1):
        print(
            f"[{i}/{len(configs)}] {cfg['name']}: "
            f"subset={cfg['subset_of_weights']} noise={cfg['noise_std']} prior={cfg['prior_std']} "
            f"step={cfg['initial_step_size']:.1e} L={cfg['leapfrog_steps']} "
            f"reduce_mean={cfg['reduce_output_mean']}"
        )
        try:
            row = _run_one(
                cfg,
                model_path=model_path,
                train_data=train_data,
                test_data=test_data,
                hmc_indices=hmc_indices,
                eval_indices=eval_indices,
                seed=args.seed,
                max_eval_draws=args.max_eval_draws,
                device=device,
            )
            rows.append(row)
            print(
                "  -> rmse={rmse:.4f} nll={nll:.4f} cov2={cov2:.3f} "
                "acc={acceptance_rate:.3f} ess_min={ess_min:.1f} "
                "draws={n_draws} sec={elapsed_sec:.1f}".format(**row)
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            rows.append({"name": cfg["name"], "error": msg})
            print(f"  -> ERROR: {msg}")

    ok_rows = [r for r in rows if "error" not in r]
    ok_rows_sorted = sorted(
        ok_rows,
        key=lambda r: (
            float("inf") if not np.isfinite(r["rmse"]) else r["rmse"],
            float("inf") if not np.isfinite(r["nll"]) else r["nll"],
        ),
    )

    print("\n=== Sorted by RMSE then NLL ===")
    for r in ok_rows_sorted:
        print(
            f"{r['name']:<28} rmse={r['rmse']:.4f} nll={r['nll']:.4f} cov2={r['cov2']:.3f} "
            f"acc={r['acceptance_rate']:.3f} ess_min={r['ess_min']:.1f} subset={r['subset_of_weights']}"
        )

    payload = {
        "seed": args.seed,
        "baseline_rmse": baseline_rmse,
        "num_hmc_data": args.num_hmc_data,
        "num_eval_data": len(eval_indices),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved tuning results to: {args.out_json}")


if __name__ == "__main__":
    main()
