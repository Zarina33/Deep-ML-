"""
escriptive Statistics Calculator
Easy
Statistics


Write a Python function to calculate various descriptive statistics metrics for a given dataset. The function should take a list or NumPy array of numerical values and return a dictionary containing:
mean: Average of all values
median: Middle value when sorted
mode: Most frequently occurring value
variance: Population variance (divide by N)
standard_deviation: Square root of variance
25th_percentile, 50th_percentile, 75th_percentile: Quartile values
interquartile_range: Difference between 75th and 25th percentiles (IQR)
Example:
Input:
[1, 2, 2, 3, 4, 4, 4, 5]
Output:
{'mean': 3.125, 'median': 3.5, 'mode': 4, 'variance': 1.6094, 'standard_deviation': 1.2686, ...}
Reasoning:
Mean = (1+2+2+3+4+4+4+5)/8 = 3.125. Median = average of 4th and 5th values = (3+4)/2 = 3.5. Mode = 4 (appears 3 times, most frequent). Variance and standard deviation measure spread around the mean. Percentiles divide the sorted data into quarters.

"""

import numpy as np
from collections import Counter

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate descriptive statistics for a given dataset.

    Args:
        data: A list or NumPy array of numerical values

    Returns:
        A dictionary containing key statistical metrics
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=float)

    if data.size == 0:
        raise ValueError("Dataset cannot be empty.")

    # Central tendency
    mean   = np.mean(data)
    median = np.median(data)

    # Mode — pick the smallest value in case of a tie
    counts   = Counter(data.tolist())
    max_freq = max(counts.values())
    mode     = min(k for k, v in counts.items() if v == max_freq)

    # Spread — population variance/std (ddof=0, divides by N)
    variance           = np.var(data)
    standard_deviation = np.std(data)

    # Percentiles & IQR (NumPy default: linear interpolation)
    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50)
    p75 = np.percentile(data, 75)
    iqr = p75 - p25

    return {
        "mean":                round(float(mean), 4),
        "median":              round(float(median), 4),
        "mode":                int(mode) if float(mode).is_integer() else mode,
        "variance":            round(float(variance), 4),
        "standard_deviation":  round(float(standard_deviation), 4),
        "25th_percentile":     round(float(p25), 4),
        "50th_percentile":     round(float(p50), 4),
        "75th_percentile":     round(float(p75), 4),
        "interquartile_range": round(float(iqr), 4),
    }