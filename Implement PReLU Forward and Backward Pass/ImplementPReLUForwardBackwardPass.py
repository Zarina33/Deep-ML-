"""
Implement PReLU Forward and Backward Pass
Medium
Deep Learning


Implement the forward and backward passes of the PReLU (Parametric ReLU) activation function. Unlike Leaky ReLU which uses a fixed slope for negative inputs, PReLU treats the negative slope as a learnable parameter updated via gradient descent. Your task is to implement both the forward computation and the backward pass that computes gradients with respect to the input and the slope parameter alpha.
Example:
Input:
prelu_forward(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), alpha=0.1)
prelu_backward(np.array([2.0, -1.0, 0.0, -3.0]), alpha=0.25, grad_output=np.array([1.0, 1.0, 1.0, 1.0]))
Output:
[-0.2, -0.1, 0.0, 1.0, 2.0]

grad_x = [1.0, 0.25, 0.25, 0.25], grad_alpha = -4.0
Reasoning:
Forward pass: Each element is passed through PReLU(x_i) = x_i if x_i > 0, else alpha * x_i. So y_0 = 2.0, y_1 = 0.25 * -1.0 = -0.25, y_2 = 0.5, and y_3 = 0.25 * -3.0 = -0.75. Backward pass - grad_x: Each element uses dL/dx_i = dL/dy_i * 1 if x_i > 0, else dL/dy_i * alpha. With grad_output all ones: grad_x_0 = 1 * 1 = 1.0, grad_x_1 = 1 * 0.25 = 0.25, grad_x_2 = 1 * 1 = 1.0, grad_x_3 = 1 * 0.25 = 0.25. Backward pass - grad_alpha: Sum dL/dy_i * x_i over elements where x_i <= 0. Only x_1 = -1.0 and x_3 = -3.0 qualify, giving grad_alpha = 1 * (-1.0) + 1 * (-3.0) = -4.0. The positive elements contribute nothing because y = x there has no dependence on alpha.

"""

import numpy as np

def prelu_forward(x: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

def prelu_backward(x: np.ndarray, alpha: float, grad_output: np.ndarray) -> tuple[np.ndarray, float]:
    grad_x = grad_output * np.where(x > 0, 1.0, alpha)
    grad_alpha = float(np.sum(grad_output * np.where(x <= 0, x, 0.0)))
    return grad_x, grad_alpha