"""
Linear Regression Using Normal Equation
Easy
Machine Learning


Write a Python function that performs linear regression using the normal equation. The function should take a matrix X (features) and a vector y (target) as input, and return the coefficients of the linear regression model. Round your answer to four decimal places, -0.0 is a valid result for rounding a very small number.
Example:
Input:
X = [[1, 1], [1, 2], [1, 3]], y = [1, 2, 3]
Output:
[0.0, 1.0]
Reasoning:
The linear model is y = 0.0 + 1.0*x, perfectly fitting the input data.

"""

import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.array(X)
    y = np.array(y)

    X_trans = X.T
    XtX = X_trans @ X
    XtX_inv = np.linalg.inv(XtX)

    Xty = X_trans @ y

    theta = XtX_inv @ Xty

    theta = np.atleast_1d(theta)
    return [round(t,4) for t in theta]