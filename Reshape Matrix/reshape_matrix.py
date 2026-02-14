"""
RESHAPE MATRIX

Write a Python function that reshapes a given matrix into a specified shape. if it cant be reshaped return back an empty list []
Example:
Input:
a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)
Output:
[[1, 2], [3, 4], [5, 6], [7, 8]]
Reasoning:
The given matrix is reshaped from 2x4 to 4x2.
"""

import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    rows = len(a)
    cols = len(a[0])

    if rows * cols != new_shape[0] * new_shape[1]:
        return []
    
    arr = []
    for i in range(rows):
        for j in range(cols):
            arr.append(a[i][j])

    reshaped = []
    index = 0

    for i in range(new_shape[0]):
        res = []
        for j in range(new_shape[0]):
            res.append(res[i][j])
        reshaped.append(res)
    return reshaped 
        
