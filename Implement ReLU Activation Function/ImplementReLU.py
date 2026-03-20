"""
Implement ReLU Activation Function
Easy
Deep Learning


Write a Python function relu that implements the Rectified Linear Unit (ReLU) activation function. The function should take a single float as input and return the value after applying the ReLU function. The ReLU function returns the input if it's greater than 0, otherwise, it returns 0.
Example:
Input:
print(relu(0)) 
print(relu(1)) 
print(relu(-1))
Output:
0
1
0


"""

def relu(z:float):
    res = max(0,z)
    return res