"""
Binomial Distribution Probability
Medium
Probability


Write a Python function to calculate the probability of achieving exactly k successes in n independent Bernoulli trials, each with probability p of success, using the Binomial distribution formula.
Example:
Input:
n = 6, k = 2, p = 0.5
Output:
0.23438
Reasoning:
We want the probability of getting exactly 2 successes in 6 trials with 50% success rate. The binomial coefficient C(6,2) = 15, and the probability calculation gives 15 × 0.25 × 0.0625 = 0.23438.

"""

import math

def binomial_probability(n: int, k: int, p: float) -> float:
    binomal = math.comb(n,k)
    success = p ** k
    failure = (1 - p)** (n - k)
    result = failure * binomal * success

    