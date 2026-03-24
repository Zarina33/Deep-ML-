"""
Calculate Jaccard Index for Binary Classification
Easy
Machine Learning

Task: Implement the Jaccard Index
Your task is to implement a function jaccard_index(y_true, y_pred) that calculates the Jaccard Index, a measure of similarity between two binary sets. The Jaccard Index is widely used in binary classification tasks to evaluate the overlap between predicted and true labels.
Your Task:
Implement the function jaccard_index(y_true, y_pred) to:
Calculate the Jaccard Index between the arrays y_true and y_pred.
Return the Jaccard Index as a float value.
Ensure the function handles cases where:
There is no overlap between y_true and y_pred.
Both arrays contain only zeros (edge cases).
The Jaccard Index is defined as:
Jaccard Index
=
Number of elements in the intersection of 
y
true
 and 
y
pred
Number of elements in the union of 
y
true
 and 
y
pred
Jaccard Index= 
Number of elements in the union of y 
true
​	
  and y 
pred
​	
 
Number of elements in the intersection of y 
true
​	
  and y 
pred
​	
 
​	
 
Where:
y
true
y 
true
​	
  and 
y
pred
y 
pred
​	
  are binary arrays of the same length, representing true and predicted labels.
The result ranges from 0 (no overlap) to 1 (perfect overlap).
Example:
Input:
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])
print(jaccard_index(y_true, y_pred))
Output:
0.75
Reasoning:
The Jaccard Index is calculated as 3 / 4 = 0.75, indicating a 75% overlap between the true and predicted labels.

"""

import numpy as np

def jaccard_index(y_true, y_pred):
    intersection = np.sum((y_true == 1) & (y_pred == 1))  # TP
    union        = np.sum((y_true == 1) | (y_pred == 1))  # TP + FP + FN
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)