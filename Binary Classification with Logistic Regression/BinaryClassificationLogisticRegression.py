"""
Binary Classification with Logistic Regression
Easy
Machine Learning


Implement the prediction function for binary classification using Logistic Regression. Your task is to compute class probabilities using the sigmoid function and return binary predictions based on a threshold of 0.5.
Example:
Input:
predict_logistic(np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]]), np.array([1, 1]), 0)
Output:
[1 1 0 0]
Reasoning:
Each sample's linear combination is computed using 
z = Xw + b
The sigmoid function is applied, and the output is thresholded at 0.5, resulting in binary predictions.



"""

import torch

def predict_logistic(X: torch.Tensor, weights: torch.Tensor, bias: float) -> torch.Tensor:

    z = X @ weights + bias
    prob = torch.sigmoid(z)
    res = (prob >=0.5).init()

    return res.numpy()
