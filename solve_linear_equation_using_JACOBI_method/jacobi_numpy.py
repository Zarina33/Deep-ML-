"""
Solve Linear Equations using Jacobi Method
Medium
Linear Algebra


Write a Python function that uses the Jacobi method to solve a system of linear equations given by Ax = b. The function should iterate n times, rounding each intermediate solution to four decimal places, and return the approximate solution x.
Initialize the solution vector x to all zeros before beginning the iterations.
Example:
Input:
A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]], b = [-1, 2, 3], n=2
Output:
[0.146, 0.2032, -0.5175]
Reasoning:
The Jacobi method iteratively solves each equation for x[i] using the formula x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i)), where a_ii is the diagonal element of A and a_ij are the off-diagonal elements.


"""
import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	num_eq = len(A)
	x = [0.0] * num_eq

	for iteration in range(n):
		x_new = [0.0] * num_eq
		for i in range(num_eq):
			suma = b[i]
			for j in range(num_eq):
				if i != j:
					suma -= A[i][j] * x[j]
			x_new[i] = round(suma / A[i][i], 4)
		x = x_new
	return x
