"""
Implement Lasso Regression using ISTA
Medium
Machine Learning
"""
import numpy as np

def soft_threshold(w, lambda_):
    """S(w, λ) = sign(w) * max(|w| - λ, 0)"""
    return np.sign(w) * np.maximum(np.abs(w) - lambda_, 0)

def l1_regularization_gradient_descent(X, y, alpha=0.1, learning_rate=0.01, max_iter=1000):
    n, p = X.shape
    w = np.zeros(p)
    b = 0.0

    for _ in range(max_iter):
        # Forward pass
        y_hat = X @ w + b

        # Residuals
        residuals = y_hat - y  # shape (n,)

        # --- Step 1: Gradient of MSE loss ---
        grad_w = (1 / n) * (X.T @ residuals)   # shape (p,)
        grad_b = (1 / n) * np.sum(residuals)    # scalar

        # --- Step 2: Gradient descent step (on smooth part) ---
        w_half = w - learning_rate * grad_w
        b      = b - learning_rate * grad_b

        # --- Step 3: Proximal step — soft-threshold (handles L1 penalty) ---
        # λ = alpha * learning_rate  (scales penalty to step size)
        w = soft_threshold(w_half, alpha * learning_rate)

        # Note: bias is NOT regularized (standard practice)

    return w, b
