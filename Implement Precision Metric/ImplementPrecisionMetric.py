"""
Implement Precision Metric
Easy
Machine Learning


Write a Python function precision that calculates the precision metric given two numpy arrays: y_true and y_pred. The y_true array contains the true binary labels, and the y_pred array contains the predicted binary labels. Precision is defined as the ratio of true positives to the sum of true positives and false positives.
Example:
Input:
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

result = precision(y_true, y_pred)
print(result)
Output:
1.0
"""

import numpy as np

def precision(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    if (TP + FP) < 0:
        return 0.0
    else:
        precision = TP / (TP + FP)
    return precision