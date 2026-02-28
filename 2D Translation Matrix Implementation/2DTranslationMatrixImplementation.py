"""
2D Translation Matrix Implementation
Medium
Linear Algebra


Task: Implement a 2D Translation Matrix
Your task is to implement a function that applies a 2D translation matrix to a set of points. A translation matrix is used to move points in 2D space by a specified distance in the x and y directions.
Write a function translate_object(points, tx, ty) where points is a list of [x, y] coordinates and tx and ty are the translation distances in the x and y directions, respectively.
The function should return a new list of points after applying the translation matrix.

"""
import numpy as np
def translate_object(points, tx, ty):
    T = np.array([
        [1,0,ty],
        [0,1,tx],
        [0,0,1]
        ])
    points = np.array(points)
    ones = np.ones((len(points),1))
    points_h = np.hstack([points,ones])

    translated = (T @ points_h.T).T

    return translated[:, :2].tolist()