"""
Softmax Activation Function Implementation
Easy
Deep Learning


Write a Python function that computes the softmax activation for a given list of scores. The function should handle numerical stability by preventing overflow when exponentiating large values. Return the softmax values as a list of floats.
Example:
Input:
scores = [1, 2, 3]
Output:
[0.0900, 0.2447, 0.6652]
Reasoning:
The softmax function converts a list of values into a probability distribution. The probabilities are proportional to the exponential of each element divided by the sum of the exponentials of all elements in the list.

"""

import numpy as np

def softmax(scores):
    scores = np.array(scores, dtype = float)
    scores -= np.max(scores)

    exp_scores = np.exp(scores)
    res = exp_scores / np.sum(exp_scores)

    return [round(float(x), 4) for x in res]
