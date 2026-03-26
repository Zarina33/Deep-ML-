"""
Implement AdaBoost Fit Method
Hard
Machine Learning


Write a Python function adaboost_fit that implements the fit method for an AdaBoost classifier. The function should take in a 2D numpy array X of shape (n_samples, n_features) representing the dataset, a 1D numpy array y of shape (n_samples,) representing the labels, and an integer n_clf representing the number of classifiers. The function should initialize sample weights, find the best thresholds for each feature, calculate the error, update weights, and return a list of classifiers with their parameters.
Example:
Input:
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 1, -1, -1])
    n_clf = 3

    clfs = adaboost_fit(X, y, n_clf)
    print(clfs)
Output:
(example format, actual values may vary):
    # [{'polarity': 1, 'threshold': 2, 'feature_index': 0, 'alpha': 0.5},
    #  {'polarity': -1, 'threshold': 3, 'feature_index': 1, 'alpha': 0.3},
    #  {'polarity': 1, 'threshold': 4, 'feature_index': 0, 'alpha': 0.2}]
Reasoning:
The function fits an AdaBoost classifier on the dataset X with the given labels y and number of classifiers n_clf. It returns a list of classifiers with their parameters, including the polarity, threshold, feature index, and alpha values


"""

import numpy as np
import math

def adaboost_fit(X, y, n_clf):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, (1 / n_samples))
    clfs = []

    for _ in range(n_clf):
        clf = {
            'polarity': 1,
            'threshold': None,
            'feature_index': None,
            'alpha': None
        }
        min_error = float('inf')

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.where(feature_values < threshold, -polarity, polarity)  # ✅
                    error = np.sum(w[predictions != y])

                    if error < min_error:
                        min_error = error
                        clf['polarity']      = polarity
                        clf['threshold']     = threshold
                        clf['feature_index'] = feature_index

        eps = 1e-10
        clf['alpha'] = 0.5 * math.log((1 - min_error + eps) / (min_error + eps))

        feature_values = X[:, clf['feature_index']]
        predictions = np.where(feature_values < clf['threshold'], -clf['polarity'], clf['polarity'])  # ✅

        w *= np.exp(-clf['alpha'] * y * predictions)
        w /= np.sum(w)

        clfs.append(clf)

    return clfs
