"""
Implement Recall Metric in Binary Classification
Easy
Machine Learning


Task: Implement Recall in Binary Classification
Your task is to implement the recall metric in a binary classification setting. Recall is a performance measure that evaluates how effectively a machine learning model identifies positive instances from all the actual positive cases in a dataset.
You need to write a function recall(y_true, y_pred) that calculates the recall metric. The function should accept two inputs:
y_true: A list of true binary labels (0 or 1) for the dataset.
y_pred: A list of predicted binary labels (0 or 1) from the model.
Your function should return the recall value as a float. If the denominator (TP + FN) is zero, return 0.0 to avoid division by zero.


"""

import numpy as np

def recall(y_true,y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    recall = TP / (TP + FN)
    if (TP + FN) < 0:
        return 0.0
    else:
        recall = TP / (TP + FN)
    return recall

