"""
Principal Component Analysis (PCA) Implementation
Medium
Machine Learning


Write a Python function that performs Principal Component Analysis (PCA) from scratch. The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. The function should standardize the dataset, compute the covariance matrix, find the eigenvalues and eigenvectors, and return the principal components (the eigenvectors corresponding to the largest eigenvalues). The function should also take an integer k as input, representing the number of principal components to return.
Sign Convention: Eigenvectors can point in either direction (if v is valid, so is -v). To ensure consistent results across different environments, apply this rule: for each eigenvector, find its first element with absolute value > 1e-10; if that element is negative, multiply the entire eigenvector by -1.
Note: Use np.linalg.eigh for eigendecomposition since covariance matrices are symmetric. This provides more numerically stable and consistent results than np.linalg.eig.
Example:
Input:
data = np.array([[1, 2], [3, 4], [5, 6]]), k = 1
Output:
[[0.7071], [0.7071]]
Reasoning:
The data lies perfectly along a diagonal line (y = x + 1). After standardization, the direction of maximum variance is along [1, 1] (normalized to [0.7071, 0.7071]). This single principal component captures 100% of the variance.


Algorithm od solution:
1)Find mean
2)Find std
3)Find standartization
4)Find covarization matrix
5)Eigendecomposition
6)Sort by ascending
7)Top - k
8)Sign convention
"""

import numpy as np

def pca(data: np.ndarray, k: int) -> list[list[float]]:
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0,ddof=1)
    X = (data - mean) / std

    n = X.shape[0]
    cov = (X.T @ X) / (n - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    top_k = eigenvectors[:, :k]
    for i in range(top_k.shape[1]):
        vec = top_k[:, i]
        # первый элемент с |x| > 1e-10
        for val in vec:
            if abs(val) > 1e-10:
                if val < 0:
                    top_k[:, i] *= -1
                break

  
    return np.round(top_k.T, 4).tolist()