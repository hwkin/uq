import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class ConformalRegressor:
    def __init__(self):
        """
        Initialize the ConformalRegressor for multi-output regression.
        """
        self.q = None
        self.tau = None
        
    def fit(self, y_true, y_pred, sigma, alpha=0.05):
        """
        Calibrates the conformal predictor using the provided calibration set.
        
        Args:
            y_true: Ground truth values, shape (m, n)
            y_pred: Predicted values, shape (m, n)
            sigma: Uncertainty estimates (e.g., standard deviation), shape (m, n)
            alpha: Significance level (default 0.05 for 95% coverage)
        """
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sigma = np.asarray(sigma)
        
        # Check shapes
        if y_true.shape != y_pred.shape or y_true.shape != sigma.shape:
            raise ValueError("Shapes of y_true, y_pred, and sigma must match.")
            
        m = y_true.shape[0]
        
        # 1. Score Function: Normalized Residual
        # Score_i = max_j (|y_true_ij - y_pred_ij| / sigma_ij)
        # We add a small epsilon to sigma to avoid division by zero if necessary, 
        # though user context implies valid sigma.
        safe_sigma = np.maximum(sigma, 1e-12)
        
        # Calculate normalized residuals per dimension
        norm_residuals = np.abs(y_true - y_pred) / safe_sigma
        
        # Take the maximum across all n dimensions (joint coverage)
        scores = np.max(norm_residuals, axis=1)
        
        # 2. Calibration: Calculate standard quantile q
        # We calculate the (1-alpha) quantile of the scores.
        # To guarantee coverage on finite samples: q_level = ceil((m+1)(1-alpha)) / m
        q_level = np.ceil((m + 1) * (1 - alpha)) / m
        # Clip to 1.0 in case of small sample size or high (1-alpha)
        q_level = min(q_level, 1.0)
        
        # 'higher' method is often used for conservative guarantees
        self.q = np.quantile(scores, q_level, method='higher')
        
        # 3. OOD Detection Threshold
        # Threshold tau based on the 99th percentile of the max sigma seen during calibration.
        max_sigmas = np.max(sigma, axis=1)
        self.tau = np.percentile(max_sigmas, 99)
        
    def predict(self, y_pred, sigma):
        """
        Generates prediction intervals and detects OOD samples.
        
        Args:
            y_pred: Predicted values for new samples (k, n)
            sigma: Uncertainty estimates for new samples (k, n)
            
        Returns:
            lower_bound: Lower bounds of the prediction intervals (k, n)
            upper_bound: Upper bounds of the prediction intervals (k, n)
            is_ood: Boolean array indicating if samples are OOD (k,)
        """
        if self.q is None or self.tau is None:
            raise RuntimeError("ConformalRegressor must be fit before calling predict.")
            
        y_pred = np.asarray(y_pred)
        sigma = np.asarray(sigma)
        
        # Calculate bounds
        # Interval is [y_pred - q * sigma, y_pred + q * sigma]
        # This applies the joint coverage factor q to all dimensions scaling by local sigma
        margin = self.q * sigma
        lower_bound = y_pred - margin
        upper_bound = y_pred + margin
        
        # OOD Detection
        # Check if the maximum uncertainty for a sample exceeds the threshold tau
        max_sigmas = np.max(sigma, axis=1)
        is_ood = max_sigmas > self.tau
        
        return lower_bound, upper_bound, is_ood
    
def den_samples(no, device, model_path, x_branch_eval=None, x_trunk_eval=None):
    preds_eval_list = []
    with torch.no_grad():
        for paths in model_path:  # model is a list of paths for models.
            m = torch.load(paths, weights_only=False).to(device)
            if no == 'deeponet':
                x_b = torch.from_numpy(x_branch_eval).float().to(device)
                x_t = torch.from_numpy(x_trunk_eval).float().to(device)
                pred = m.predict(x_b, x_t)
            elif no == 'pcanet':
                x_tensor = torch.from_numpy(x_branch_eval).float().to(device)
                pred = m.predict(x_tensor)
            elif no == 'fno':
                pred = m.predict(x_branch_eval)
                pred = pred.reshape(pred.shape[0], -1)
            else:
                raise NotImplementedError(f"Model type '{no}' not implemented.")
            preds_eval_list.append(pred.cpu().numpy())
    preds_eval = np.stack(preds_eval_list)
    mean_eval = np.mean(preds_eval, axis=0)
    std_eval = np.std(preds_eval, axis=0)
    return mean_eval, std_eval