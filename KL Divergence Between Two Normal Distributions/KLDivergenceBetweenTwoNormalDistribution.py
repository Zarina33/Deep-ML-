"""
KL Divergence Between Two Normal Distributions
Easy
Deep Learning


Task: Implement KL Divergence Between Two Normal Distributions
Your task is to compute the Kullback Leibler (KL) divergence between two normal distributions. KL divergence measures how one probability distribution differs from a second, reference probability distribution.
Write a function kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q) that calculates the KL divergence between two normal distributions.
The function should return the KL divergence as a floating point number.
Example:
Input:
mu_p = 0.0
sigma_p = 1.0
mu_q = 1.0
sigma_q = 1.0

print(kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q))
Output:
0.5
Reasoning:
The KL divergence between the normal distributions ( P ) and ( Q ) with parameters ( \mu_P = 0.0 ), ( \sigma_P = 1.0 ) and ( \mu_Q = 1.0 ), ( \sigma_Q = 1.0 ) is 0.5.
"""

import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    # KL(P || Q) = log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²) / (2σ_q²) - 1/2
    term1 = np.log(sigma_q / sigma_p)
    term2 = (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2)
    term3 = 0.5
    
    return float(term1 + term2 - term3)