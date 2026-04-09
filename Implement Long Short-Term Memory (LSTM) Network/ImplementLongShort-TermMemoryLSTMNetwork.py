"""
Implement Long Short-Term Memory (LSTM) Network
Medium
Deep Learning


Task: Implement Long Short-Term Memory (LSTM) Network
Your task is to implement an LSTM network that processes a sequence of inputs and produces the final hidden state and cell state after processing all inputs.
Write a class LSTM with the following methods:
__init__(self, input_size, hidden_size): Initializes the LSTM with random weights and zero biases.
forward(self, x, initial_hidden_state, initial_cell_state): Processes a sequence of inputs and returns the hidden states at each time step, as well as the final hidden state and cell state.
The LSTM should compute the forget gate, input gate, candidate cell state, and output gate at each time step to update the hidden state and cell state.
Example:
Input:
input_sequence = np.array([[1.0], [2.0], [3.0]])
initial_hidden_state = np.zeros((1, 1))
initial_cell_state = np.zeros((1, 1))

lstm = LSTM(input_size=1, hidden_size=1)
outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)

print(final_h)
Output:
[[0.73698596]] (approximate)
Reasoning:
The LSTM processes the input sequence [1.0, 2.0, 3.0] and produces the final hidden state [0.73698596].

"""

import numpy as np

class LSTM:
    def __init__(self,input_size,hidden_size):
        #определяем слои 
        self.input_size = input_size
        self.hidden_size = hidden_size


        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.rand(input_size, input_size + hidden_size)
        self.Wo = np.random.randn(input_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size,1))
        self.bi = np.zeros((hidden_size,1))
        self.bc = np.zeros((input_size,1))
        self.bo = np.zeros((input_size, 1))
        

        def forward(self,x, initial_hidden_state, initial_cell_state):
            h = initial_cell_state
            c = initial_cell_state
            
            hidden_states = []

            for t in range(len(x)):
                xt = x[t].reshape(-1,1)
                combined = np.vstack([h,xt])

                f = self._sigmoid(self.Wf @ combined + self.bf) #что забыть
                i = self._sigmoid(self.Wi @ combined + self.bi) #что сохранить 
                g = np.tanh(self.Wc @ combined + self.bc) #что именно
                o = self._sigmoid(self.Wo @ combined + self.bo) #что выдать

                c = f * c * i * g
                h = o * np.tanh

                hidden_states.append(h.T)
            return hidden_states, h.T , c.T
        
        def _sigmoid(self,z):
            return 1/(1 + np.exp(-z))