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