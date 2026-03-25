"""
Pegasos Kernel SVM Implementation
Hard
Machine Learning


Write a Python function that implements a deterministic version of the Pegasos algorithm to train a kernel SVM classifier from scratch. The function should take a dataset (as a 2D NumPy array where each row represents a data sample and each column represents a feature), a label vector (1D NumPy array where each entry corresponds to the label of the sample), and training parameters such as the choice of kernel (linear or RBF), regularization parameter (lambda), and the number of iterations. Note that while the original Pegasos algorithm is stochastic (it selects a single random sample at each step), this problem requires using all samples in every iteration (i.e., no random sampling). The function should perform binary classification and return the model's alpha coefficients as a list and bias as a float.
Example:
Input:
data = np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), labels = np.array([1, 1, -1, -1]), kernel = 'linear', lambda_val = 0.01, iterations = 100
Output:
([100.0, 0.0, -100.0, -100.0], -937.4755)
Reasoning:
Using the linear kernel, the Pegasos algorithm iteratively updates the alpha coefficients and bias. Points that violate the margin constraint (y * f(x) < 1) get their alphas updated. After 100 iterations, the first and last two points become support vectors with large alpha values, while the second point (which is well-classified) has alpha = 0.

"""

import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    n = len(data)
    alphas = np.zeros(n)
    bias = 0.0

    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if kernel == 'linear':
                K[i][j] = data[i] @ data[j]
            elif kernel == 'rbf':
                diff = data[i] - data[j]
                K[i][j] = np.exp(-np.sum(diff ** 2) / (2 * sigma ** 2))

    for t in range(1, iterations + 1):
        eta = 1 / (lambda_val * t)
        for i in range(n):
            f_i = np.sum(alphas * labels * K[:, i]) + bias
            if labels[i] * f_i < 1:
                alphas[i] += eta * (labels[i] - lambda_val * alphas[i])  # ✅
                bias += eta * labels[i]
            # ✅ NO else — не трогаем невиолирующие точки

    return (
        [round(float(a), 4) for a in alphas],
        round(float(bias), 4)
    )