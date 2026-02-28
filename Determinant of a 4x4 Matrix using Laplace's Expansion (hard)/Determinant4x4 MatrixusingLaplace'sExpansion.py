"""
Determinant of a 4x4 Matrix using Laplace's Expansion (hard)
Hard
Linear Algebra


Write a Python function that calculates the determinant of a 4x4 matrix using Laplace's Expansion method. The function should take a single argument, a 4x4 matrix represented as a list of lists, and return the determinant of the matrix. The elements of the matrix can be integers or floating-point numbers. Implement the function recursively to handle the computation of determinants for the 3x3 minor matrices.
Example:
Input:
a = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
Output:
0
Reasoning:
Using Laplace's Expansion, the determinant of a 4x4 matrix is calculated by expanding it into minors and cofactors along any row or column. Given the symmetrical and linear nature of this specific matrix, its determinant is 0. The calculation for a generic 4x4 matrix involves more complex steps, breaking it down into the determinants of 3x3 matrices.

"""

def determinant_4x4(matrix: list[list[int|float]]) -> float:
    # База рекурсии — 2×2
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for j in range(len(matrix)):
        # Шаг 1: вычеркиваем строку 0 и столбец j
        minor = [
            [matrix[i][k] for k in range(len(matrix)) if k != j]
            for i in range(1, len(matrix))  # строки кроме 0
        ]
        
        # Шаг 2: знак чередуется + - + -
        sign = (-1) ** j
        
        # Шаг 3: добавляем к определителю
        det += matrix[0][j] * sign * determinant_4x4(minor)
    
    return det