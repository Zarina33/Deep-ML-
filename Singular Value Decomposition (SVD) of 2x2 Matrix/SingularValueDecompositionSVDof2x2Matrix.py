"""
Singular Value Decomposition (SVD) of 2x2 Matrix
Hard
Linear Algebra


Write a Python function that computes an approximate Singular Value Decomposition (SVD) of a real 2x2 matrix using one Jacobi rotation.
Input:
A: a NumPy array of shape (2, 2)
Rules:
You may use basic NumPy operations (matrix multiplication, transpose, element-wise math, etc.)
Do NOT call numpy.linalg.svd or any other high-level SVD routine
Use a single Jacobi rotation step (no iterative refinements)
Return: A tuple (U, S, Vt) where:
U is a 2x2 orthogonal matrix (left singular vectors)
S is a length-2 NumPy array containing the singular values
Vt is the transpose of the right singular vector matrix V
The decomposition should approximately satisfy: A = U @ diag(S) @ Vt
Example:
Input:
A = np.array([[2, 1], [1, 2]])
Output:
U ≈ [[0.707, -0.707], [0.707, 0.707]]
S = [3.0, 1.0]
Vt ≈ [[0.707, 0.707], [-0.707, 0.707]]
Reasoning:
The symmetric matrix [[2,1],[1,2]] has eigenvalues 3 and 1. Since it's symmetric, the SVD simplifies: the singular values equal the absolute eigenvalues, and U, V are related to the eigenvectors. The decomposition satisfies A = U @ diag(S) @ Vt.
"""

import numpy as np

def svd_2x2_jacobi(A):
    # Шаг 1: AᵀA
    AtA = A.T @ A
    
    # Шаг 2: угол Jacobi rotation
    if abs(AtA[0,1]) < 1e-10:
        theta = 0.0
    else:
        theta = 0.5 * np.arctan2(2 * AtA[0,1], AtA[0,0] - AtA[1,1])
    
    # Шаг 3: матрица поворота V
    c = np.cos(theta)
    s = np.sin(theta)
    V = np.array([[c, -s],
                  [s,  c]])
    
    # Шаг 4: singular values из диагонали
    diag = V.T @ AtA @ V
    S = np.sqrt(np.abs(np.diag(diag)))
    
    # Сортируем по убыванию
    if S[0] < S[1]:
        S = S[::-1]
        V = V[:, ::-1]
    
    # Шаг 5: U = AV / S
    U = np.zeros((2, 2))
    for i in range(2):
        if S[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / S[i]
    
    # Sign convention
    for i in range(2):
        for val in U[:, i]:
            if abs(val) > 1e-10:
                if val < 0:
                    U[:, i] *= -1
                    V[:, i] *= -1
                break
    
    return U, S, V.T