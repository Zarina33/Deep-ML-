"""
Implement Gradient Descent Variants with MSE Loss
Medium
Machine Learning

In this problem, you need to implement a single function that can perform three variants of gradient descent: Stochastic Gradient Descent (SGD), Batch Gradient Descent, and Mini-Batch Gradient Descent using Mean Squared Error (MSE) as the loss function. The function will take an additional parameter to specify which variant to use.
Requirements
Do not shuffle the data; process samples in their original order (index 0, 1, 2, ...)
For Batch GD: use all samples to compute a single gradient update per epoch
For Stochastic GD: iterate through each sample sequentially (i.e., process sample 0, then 1, then 2, etc.) â not randomly selected
For Mini-Batch GD: form batches from consecutive samples without overlap (e.g., for batch_size=2: first batch uses indices [0,1], second batch uses [2,3], etc.)
The n_epochs parameter specifies how many complete passes through the dataset to perform
For each epoch, process all samples according to the specified method
"""

import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    m = len(y)
    theta = weights.copy()

    for epoch in range(n_epochs):

        if method == 'batch':
            y_pred = X @ theta
            error = y_pred - y
            gradient = 2 * X.T @ error / m
            theta = theta - learning_rate * gradient

        elif method =='stochastic':
            for i in range(m):
                y_pred = X[i] @ theta
                error = y_pred - y[i] 
                gradient = 2 * X[i] * error
                theta = 2 * theta - learning_rate * gradient

        elif method == 'mini_batch':
            for i in range(m):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                y_pred = X_batch @ theta
                error = y_pred - y_batch
                gradient = 2 * X_batch.T @ error / batch_size
                theta = theta - learning_rate * gradient

        return theta
