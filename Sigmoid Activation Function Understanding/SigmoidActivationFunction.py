"""
Sigmoid Activation Function Understanding
Easy
Deep Learning


Write a Python function that computes the output of the sigmoid activation function given an input value z. The function should return the output rounded to four decimal places.
Example:
Input:
z = 0
Output:
0.5
Reasoning:
The sigmoid function is defined as σ(z) = 1 / (1 + exp(-z)). For z = 0, exp(-0) = 1, hence the output is 1 / (1 + 1) = 0.5.

"""

import math

def sigmoid(z: float) -> float:
	result = 1 / (1 + math.exp(-z))
	return round(result,4)