"""
Write a Python function to calculate the covariance matrix for a given set of vectors. The function should take a list of lists, where each inner list represents a feature with its observations, and return a covariance matrix as a list of lists.
Example:
Input:
[[1, 2, 3], [4, 5, 6]]
Output:
[[1.0, 1.0], [1.0, 1.0]]
Reasoning:
The covariance between the two features is calculated based on their deviations from the mean. For the given vectors, both covariances are 1.0, resulting in a symmetric covariance matrix.
"""

import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    matrix = np.array(vectors)
    n= matrix.shape[1]
    mean = matrix.mean(axis=1, keepdims=True)
    centered = matrix - mean
    cov = (centered @ centered.T) / (n - 1)
    return cov.tolist()