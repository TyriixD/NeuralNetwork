import numpy as np
from Layer import Layer_Dense
from nnfs.datasets import spiral_data


#creates a dataset
X, y = spiral_data(samples=100, classes=3)

#create a dense layer with 2 input and 3 output values
dense1 = Layer_Dense(2, 3)
#Perform a forward pass of our training data through a layer
dense1.forward(X)

#5 rows of data with 3 values each. Each of the value is the value from the 3 neurons in the dense1 layer after passing in each samples.
print(dense1.output[:5])