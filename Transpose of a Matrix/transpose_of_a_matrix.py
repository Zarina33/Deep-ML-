"""
Write a Python function that computes the transpose of a given 2D matrix. The transpose of a matrix is formed by turning its rows into columns and columns into rows. For an mÃn matrix, the transpose will be an nÃm matrix.
Example:
Input:
a = [[1, 2, 3], [4, 5, 6]]
Output:
[[1, 4], [2, 5], [3, 6]]
Reasoning:
The input is a 2×3 matrix. The transpose swaps rows and columns: the first row [1, 2, 3] becomes the first column, and the second row [4, 5, 6] becomes the second column, resulting in a 3×2 matrix.
"""


def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    if not a or not a[0]:
        return []
    rows = len(a)
    cols = len(a[0])

    final_matrix = []

    for j in range(cols):
        res = []
        for i in range(rows):
            res.append(a[i][j])
        final_matrix.append(res)
    return final_matrix
    