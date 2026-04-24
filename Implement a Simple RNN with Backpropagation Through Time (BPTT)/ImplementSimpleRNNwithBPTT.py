"""
Task: Implement a Simple RNN with Backpropagation Through Time (BPTT)
Your task is to implement a simple Recurrent Neural Network (RNN) and backpropagation through time (BPTT) to learn from sequential data. The RNN will process input sequences, update hidden states, and perform backpropagation to adjust weights based on the error gradient.
Write a class SimpleRNN with the following methods:
__init__(self, input_size, hidden_size, output_size): Initializes the RNN with random weights and zero biases.
forward(self, x): Processes a sequence of inputs and returns the hidden states and output.
backward(self, x, y, learning_rate): Performs backpropagation through time (BPTT) to adjust the weights based on the loss.
In this task, the RNN will be trained on sequence prediction, where the network will learn to predict the next item in a sequence. You should use 1/2 * Mean Squared Error (MSE) as the loss function and make sure to aggregate the losses at each time step by summing.
Example:
Input:
import numpy as np
    input_sequence = np.array([[1.0], [2.0], [3.0], [4.0]])
    expected_output = np.array([[2.0], [3.0], [4.0], [5.0]])
    # Initialize RNN
    rnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1)
    
    # Forward pass
    output = rnn.forward(input_sequence)
    
    # Backward pass
    rnn.backward(input_sequence, expected_output, learning_rate=0.01)
    
    print(output)
    
    # The output should show the RNN predictions for each step of the input sequence.
Output:
[[x1], [x2], [x3], [x4]]
Reasoning:
The RNN processes the input sequence [1.0, 2.0, 3.0, 4.0] and predicts the next item in the sequence at each step.
"""

import numpy as np


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_xh = np.random.randn(hidden_size, input_size)  * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h  = np.zeros((hidden_size, 1))
        self.b_y  = np.zeros((output_size, 1))

    def forward(self, x):
        """
        x: (T, input_size)
        Возвращает массив формы (T, output_size, 1).
        Кэширует x_t, h_t, y_t для BPTT.
        """
        T = x.shape[0]
        self.xs = {}
        self.hs = {-1: np.zeros((self.hidden_size, 1))}   # h_{-1} = 0
        self.ys = {}
        outputs = []

        for t in range(T):
            x_t = x[t].reshape(-1, 1)
            h_t = np.tanh(self.W_xh @ x_t + self.W_hh @ self.hs[t-1] + self.b_h)
            y_t = self.W_hy @ h_t + self.b_y

            self.xs[t] = x_t
            self.hs[t] = h_t
            self.ys[t] = y_t
            outputs.append(y_t)

        return np.array(outputs)

    def backward(self, x, y, learning_rate):
        """
        BPTT. Loss = sum_t 1/2 * (y_hat_t - y_t)^2.
        Предполагается, что forward(x) уже был вызван и кэш актуален.
        """
        T = x.shape[0]

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h  = np.zeros_like(self.b_h)
        db_y  = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))   # в последний шаг ничего из будущего не приходит

        for t in reversed(range(T)):
            y_target = y[t].reshape(-1, 1)

            # dL/dy_t = y_hat_t - y_t (из 1/2 * MSE)
            dy = self.ys[t] - y_target

            # выходной слой
            dW_hy += dy @ self.hs[t].T
            db_y  += dy

            # градиент по скрытому состоянию: от выхода + от следующего шага
            dh     = self.W_hy.T @ dy + dh_next
            dh_raw = (1 - self.hs[t] ** 2) * dh          # производная tanh

            # скрытый слой
            db_h  += dh_raw
            dW_xh += dh_raw @ self.xs[t].T
            dW_hh += dh_raw @ self.hs[t-1].T

            dh_next = self.W_hh.T @ dh_raw               # передаём в прошлое

        # SGD-шаг
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h  -= learning_rate * db_h
        self.b_y  -= learning_rate * db_y