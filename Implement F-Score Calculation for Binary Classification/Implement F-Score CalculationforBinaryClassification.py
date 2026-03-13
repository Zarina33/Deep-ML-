"""
Implement F-Score Calculation for Binary Classification
Easy
Machine Learning


Task: Implement F-Score Calculation for Binary Classification
Your task is to implement a function that calculates the F-Score for a binary classification task. The F-Score combines both Precision and Recall into a single metric, providing a balanced measure of a model's performance.
Write a function f_score(y_true, y_pred, beta) where:
y_true: A numpy array of true labels (binary).
y_pred: A numpy array of predicted labels (binary).
beta: A float value that adjusts the importance of Precision and Recall. When beta=1, it computes the F1-Score, a balanced measure of both Precision and Recall.
The function should return the F-Score rounded to three decimal places.
Example:
Input:
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])
beta = 1

print(f_score(y_true, y_pred, beta))
Output:
0.857
Reasoning:
The F-Score for the binary classification task is calculated using the true labels, predicted labels, and beta value.

"""

import numpy as np

def fOneScore(y_true, y_pred, beta):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    if (TP + FP) > 0 and (TP + FN) > 0:
        precission = TP / (TP + FP)
        recal = TP / (TP + FN)
    else:
        precission = 0.0
        recal = 0.0
    betta = beta **2
    fOne = (1 + betta) * ((precission * recal)/((betta * precission) + recal))
    return fOne
