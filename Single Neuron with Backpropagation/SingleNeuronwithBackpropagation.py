"""
Single Neuron with Backpropagation
Medium
Deep Learning


Write a Python function that simulates a single neuron with sigmoid activation, and implements batch backpropagation to update the neuron's weights and bias. The function should take a list of feature vectors, associated true binary labels, initial weights, initial bias, a learning rate, and the number of epochs. Use full-batch gradient descent: in each epoch, perform a single forward pass over all training examples, compute the MSE loss and its gradients averaged across the entire batch, then apply one weight and bias update per epoch. Record the MSE before each update (i.e., using the weights at the start of the epoch), so mse_values has length equal to epochs. Return the updated weights, bias, and the list of MSE values, each rounded to four decimal places.
Example:
Input:
features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]], labels = [1, 0, 0], initial_weights = [0.1, -0.2], initial_bias = 0.0, learning_rate = 0.1, epochs = 2
Output:
updated_weights = [0.1036, -0.1425], updated_bias = -0.0167, mse_values = [0.3033, 0.2942]
Reasoning:
The neuron receives feature vectors and computes predictions using the sigmoid activation. Based on the predictions and true labels, the gradients of MSE loss with respect to weights and bias are computed and used to update the model parameters across epochs.
"""

import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	
	X = np.array(features)
	y = np.array(labels)
	w = np.array(initial_weights,dtype = float)
	b = float(initial_bias)
	n = len(y)
	mse_values = []

	def sigmoid(z):
		result = 1 / (1 + np.exp(-z))
		return result
	
	for _ in range(epochs):
		#прямо обучается
		z = X @ w + b
		y_hat = sigmoid(z)

		#вычисляем ошибку перед обновлением 
		mse = np.mean((y_hat - y)**2)
		mse_values.append(round(mse,4))

		#вычисляем back propagation 
		error = y_hat - y
		grad_common = error * y_hat *(1-y_hat)
		grad_w = (2/n) * (X.T @ grad_common)
		grad_b = (2/n) * np.sum(grad_common)


		#вычисляем графдиентный спуск

		w -=learning_rate * grad_w
		b -=learning_rate * grad_b

	return(np.round(w,4), round(b,4),mse_values)