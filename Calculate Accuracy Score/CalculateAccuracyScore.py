"""
Calculate Accuracy Score
Easy
Machine Learning


Write a Python function to calculate the accuracy score of a model's predictions. The function should take in two 1D numpy arrays: y_true, which contains the true labels, and y_pred, which contains the predicted labels. It should return the accuracy score as a float.
Example:
Input:
y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1])
    output = accuracy_score(y_true, y_pred)
    print(output)
Output:
# 0.8333333333333334
"""

import numpy as np
def accuracy_score(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)
