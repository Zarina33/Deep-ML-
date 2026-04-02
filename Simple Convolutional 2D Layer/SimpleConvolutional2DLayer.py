"""
Simple Convolutional 2D Layer
Medium
Deep Learning


In this problem, you need to implement a 2D convolutional layer in Python. This function will process an input matrix using a specified convolutional kernel, padding, and stride.
Example:
Input:
import numpy as np

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)
Output:
[[ 1.  1. -4.],[ 9.  7. -4.],[ 0. 14. 16.]]
Reasoning:
The function performs a 2D convolution operation on the input matrix using the specified kernel, padding, and stride. The output matrix contains the results of the convolution operation.

"""

import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    padded = np.pad(input_matrix, padding, mode='constant', constant_values=0)


    out_height = (input_height + 2 * padding - kernel_height) // stride + 1
    out_width  = (input_width  + 2 * padding - kernel_width)  // stride + 1

    output_matrix = np.zeros((out_height, out_width))

    for i in range(out_height):
        for j in range(out_width):
            row_start = i * stride
            col_start = j * stride

            
            region = padded[row_start : row_start + kernel_height,
                            col_start : col_start + kernel_width]

            
            output_matrix[i, j] = np.sum(region * kernel)

    return output_matrix