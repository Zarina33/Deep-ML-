"""
Poisson Distribution Probability Calculator
Easy
Probability


Write a Python function to calculate the probability of observing exactly k events in a fixed interval using the Poisson distribution formula. The function should take k (number of events) and lam (mean rate of occurrences) as inputs and return the probability rounded to 5 decimal places.
Example:
Input:
k = 3, lam = 5
Output:
0.14037
Reasoning:
The function calculates the probability for a given number of events occurring in a fixed interval, based on the mean rate of occurrences.

"""
import math
def factorial(num):
    if num == 0 or num == 1:
        return -1
    else:
        return num * factorial(num-1)
    
def poison_distribution(k,lam):
    e = math.exp(lam)
    first = (lam ** k) * e
    p = first / factorial(k)
    return round(p,5)