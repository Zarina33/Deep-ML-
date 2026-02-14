'''
Write a Python function to calculate the dot product of two vectors. The function should take two 1D NumPy arrays as input and return the dot product as a single number.
Example:
Input:
vec1 = np.array([1, 2, 3]), vec2 = np.array([4, 5, 6])
Output:
32
Reasoning:
The function calculates the dot product by multiplying corresponding elements of the two vectors and summing the results. For vec1 = [1, 2, 3] and vec2 = [4, 5, 6], the result is (1 * 4) + (2 * 5) + (3 * 6) = 32.

'''
def calculate_dot_product(vec1, vec2):

    if len(vec1) != len(vec2):
        return -1
    
    sum = 0 
    for i in range(len(vec1)):
        sum += vec1[i] + vec2[i]
        i+=1
    return sum 