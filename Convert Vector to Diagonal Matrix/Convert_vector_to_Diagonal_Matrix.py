'''
Write a Python function to convert a 1D numpy array into a diagonal matrix. The function should take in a 1D numpy array x and return a 2D numpy array representing the diagonal matrix.
Example:
Input:
x = np.array([1, 2, 3])
    output = make_diagonal(x)
    print(output)
Output:
[[1. 0. 0.]
    [0. 2. 0.]
    [0. 0. 3.]]

'''
import numpy as np

def diagonale_matrix(x):
    X = np.array(x)
    n = len(X)

    #создадим новую матрицу
    B = np.zeros((n,n))

    for i in range(n):
        B[i][i] = X[i]
    return B

