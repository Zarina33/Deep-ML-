"""
Gauss-Seidel Method for Solving Linear Systems
Medium
Linear Algebra


Task: Implement the Gauss-Seidel Method
Your task is to implement the Gauss-Seidel method, an iterative technique for solving a system of linear equations (Ax = b).
The function should iteratively update the solution vector (x) by using the most recent values available during the iteration process.
Write a function gauss_seidel(A, b, n, x_ini=None) where:
A is a square matrix of coefficients,
b is the right-hand side vector,
n is the number of iterations,
x_ini is an optional initial guess for (x) (if not provided, initialize with zeros).
The function should return the approximated solution vector (x) after performing the specified number of iterations.
Assumptions:
The matrix A is diagonally dominant (ensures convergence)
All diagonal elements of A are non-zero
The system has a unique solution
Example:
Input:
A = np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]], dtype=float)
b = np.array([4, 7, 3], dtype=float)

n = 100
print(gauss_seidel(A, b, n))
Output:
# [0.2, 1.4, 0.8]  (Approximate, values may vary depending on iterations)
Reasoning:
The Gauss-Seidel method iteratively updates the solution vector (x) until convergence. The output is an approximate solution to the linear system.
Learn About topic
Contributors:
paddywardle
Contribute
Request Edit


"""

import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
    num_eq = len(A)

    if x_ini is not None:
        x = x_ini.copy()
    else:
        x = np.zeros(num_eq)

    for iteration in range(n):
        for i in range(num_eq):
            suma = b[i]
            for j in range(num_eq):
                suma -= A[i][j] * x[i]
            x[i] = suma/ A[i][i]
    return np.round(x,4)