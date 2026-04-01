"""
Single Neuron
Easy
Deep Learning


Write a Python function that simulates a single neuron with a sigmoid activation function for binary classification, handling multidimensional input features. The function should take a list of feature vectors (each vector representing multiple features for an example), associated true binary labels, and the neuron's weights (one for each feature) and bias as input. It should return the predicted probabilities after sigmoid activation and the mean squared error between the predicted probabilities and the true labels, both rounded to four decimal places.
Example:
Input:
features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], labels = [0, 1, 0], weights = [0.7, -0.4], bias = -0.1
Output:
([0.4626, 0.4134, 0.6682], 0.3349)
Reasoning:
For each input vector, the weighted sum is calculated by multiplying each feature by its corresponding weight, adding these up along with the bias, then applying the sigmoid function to produce a probability. The MSE is calculated as the average squared difference between each predicted probability and the corresponding true label.

"""
import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float): # type: ignore
    
    def sigmoid(z: float) -> float:
        return 1 / (1 + math.exp(-z))
    
    def compute_weighted_sum(feature_vector: list[float]) -> float:
        weighted_sum = bias
        for feature, weight in zip(feature_vector, weights):
            weighted_sum += feature * weight
        return weighted_sum
    
    def compute_mse(predicted: list[float], actual: list[int]) -> float:
        total_squared_error = 0.0
        for prediction, true_label in zip(predicted, actual):
            error = prediction - true_label
            total_squared_error += error ** 2
        return round(total_squared_error / len(actual), 4)
    
    
    probabilities = []
    for feature_vector in features:
        z = compute_weighted_sum(feature_vector)
        probability = round(sigmoid(z), 4)
        probabilities.append(probability)
    
    
    mse = compute_mse(probabilities, labels)
    
    return probabilities, mse