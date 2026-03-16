"""
Calculate Mean Absolute Error (MAE)
Easy
Machine Learning

Implement a function to calculate the Mean Absolute Error (MAE) between two arrays of actual and predicted values. The MAE is a metric used to measure the average magnitude of errors in a set of predictions without considering their direction.
Your function should return the MAE as a float value.
Example:
Input:
y_true = np.array([3, -0.5, 2, 7]), y_pred = np.array([2.5, 0.0, 2, 8])
Output:
0.5
Reasoning:
The MAE is the mean of absolute differences: (|3-2.5| + |-0.5-0| + |2-2| + |7-8|) / 4 = (0.5 + 0.5 + 0 + 1) / 4 = 0.5

"""
import numpy as np
def mae (y_true, y_pred):
    n = len(y_true)
    lol = 0

    for i in range(n):
        lol+= np.abs(y_true[i] - y_pred[i])
    mae = (1 /n) * lol
    return mae


