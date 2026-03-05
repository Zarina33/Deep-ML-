"""
Implement Adam Optimization Algorithm
Medium
Deep Learning


Implement the Adam (Adaptive Moment Estimation) optimization algorithm in Python. Adam is an optimization algorithm that adapts the learning rate for each parameter. Your task is to write a function adam_optimizer that updates the parameters of a given function using the Adam algorithm.
The function should take the following parameters:
f: The objective function to be optimized
grad: A function that computes the gradient of f
x0: Initial parameter values
learning_rate: The step size (default: 0.001)
beta1: Exponential decay rate for the first moment estimates (default: 0.9)
beta2: Exponential decay rate for the second moment estimates (default: 0.999)
epsilon: A small constant for numerical stability (default: 1e-8)
num_iterations: Number of iterations to run the optimizer (default: 1000)
"""

import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    
    for t in range(1, num_iterations + 1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_corr = m / (1 - beta1**t)
        v_corr = v / (1 - beta2**t)
        x = x - learning_rate * m_corr / (np.sqrt(v_corr) + epsilon)
    
    return x