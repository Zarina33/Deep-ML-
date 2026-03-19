"""
Calculate Performance Metrics for a Classification Model
Medium
Machine Learning


Task: Implement Performance Metrics Calculation
In this task, you are required to implement a function performance_metrics(actual, predicted) that computes various performance metrics for a binary classification problem. These metrics include:
Confusion Matrix
Accuracy
F1 Score
Specificity
Negative Predictive Value
The function should take in two lists:
actual: The actual class labels (1 for positive, 0 for negative).
predicted: The predicted class labels from the model.
Output
The function should return a tuple containing:
confusion_matrix: A 2x2 matrix.
accuracy: A float representing the accuracy of the model.
f1_score: A float representing the F1 score of the model.
specificity: A float representing the specificity of the model.
negative_predictive_value: A float representing the negative predictive value.
Constraints
All elements in the actual and predicted lists must be either 0 or 1.
Both lists must have the same length.
Example:
Input:
actual = [1, 0, 1, 0, 1]
predicted = [1, 0, 0, 1, 1]
print(performance_metrics(actual, predicted))
Output:
([[2, 1], [1, 1]], 0.6, 0.667, 0.5, 0.5)
Reasoning:
The function calculates the confusion matrix, accuracy, F1 score, specificity, and negative predictive value based on the input labels. The resulting values are rounded to three decimal places as required.

"""

import numpy as np
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    actual = np.array(actual)
    predicted = np.array(predicted)

    TP = np.sum((actual == 1) & (predicted == 1))
    TN = np.sum((actual == 0) & (predicted == 0))
    FP = np.sum((actual == 0) & (predicted == 1))
    FN = np.sum((actual == 1) & (predicted == 0))

    accuracy = np.sum(actual == predicted) / len(actual)

    confusion_matrix = [[int(TP), int(FN), [int(FP), int(TN)]]]

    recall = TP / (TP + FN)
    precission = TP / (TP + FP)
    f1 = 2 * (recall * precission) / (recall + precission)

    specificity = TN / (TN + FP)
    negativePredictive = TN / (TN + FN)
    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)