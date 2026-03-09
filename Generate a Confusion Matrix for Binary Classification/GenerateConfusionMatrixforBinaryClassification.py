"""
Generate a Confusion Matrix for Binary Classification
Easy
Machine Learning

Task: Generate a Confusion Matrix
Your task is to implement the function confusion_matrix(data) that generates a confusion matrix for a binary classification problem. The confusion matrix provides a summary of the prediction results on a classification problem, allowing you to visualize how many data points were correctly or incorrectly labeled.
Input:
A list of lists, where each inner list represents a pair
[y_true, y_pred] for one observation. y_true is the actual label, and y_pred is the predicted label.
Output:
A 
2
×
2
2×2 confusion matrix represented as a list of lists.

"""

from collections import Counter

def confusion_matrix(data):
    TP = TN = FP = FN = 0
    
    for y_true, y_pred in data:
        if y_true == 1 and y_pred == 1:
            TP += 1
        elif y_true == 0 and y_pred == 0:
            TN += 1
        elif y_true == 0 and y_pred == 1:
            FP += 1
        elif y_true == 1 and y_pred == 0:
            FN += 1
    
    return [[TN, FN], [FP, TP]]
