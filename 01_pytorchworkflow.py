# PyTorch Workflow
# 1. Design model (input, output size, forward pass)
# 2. Construct loss and optimizer
# 3. Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights
# 4. Evaluate
#   - no_grad
#   - save model
# 5. Load model


# What is linspace?
#Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive. That is, the value a

# What is Generalization?
# The ability of a model to predict on unseen data accurately.

import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import numpy as np
import matplotlib.pyplot as plt


# 1) Model
# Linear model f = wx + b (or y = mx + b)
# weight = b, bias = w. 
# Bias is the intercept of the line. 
# What does the bias do? It shifts the line up or down.
# What does shifting the line up or down do? It changes the output of the model.
# What does it mean to have a positive bias? what does it mean to have a negative bias?
# A positive bias means that the output of the model will be shifted up. A negative bias means that the output of the model will be shifted down.
 
# Weight is the slope of the line. 
# what does it mean to have a steep slope? what does it mean to have a shallow slope?
# A steep slope means that the output is very sensitive to the input. A shallow slope means that the output is not very sensitive to the input.
# what does it mean to have a positive slope? what does it mean to have a negative slope?
# A positive slope means that the output increases as the input increases. A negative slope means that the output decreases as the input increases.

# f = wb + b

weight = 0.5
bias = 0.3 # y = 0.5x + 0.3

start = 0
end = 50
step = 3

X = torch.arange(start, end, step).unsqueeze(1) # X is the input
y = weight * X + bias # y is the output

# The point of ML is to learn the representation of the input and how it maps to the output.
# what does the result mean?
# y = 0.5x + 0.3
# y = 0.5 * 0 + 0.3 = 0.3
# y = 0.5 * 3 + 0.3 = 1.8
# y = 0.5 * 6 + 0.3 = 3.3
# y = 0.5 * 9 + 0.3 = 4.8
# ...
# y = 0.5 * 48 + 0.3 = 24.3

# Split data into training and test sets
n_samples = X.shape[0] # 50
n_train = int(n_samples * 0.8) # 40
n_test = n_samples - n_train # 10
print(n_train)
print(n_test)

# plot predictions
plt.figure(figsize=(12, 8))
plt.scatter(X[:n_train], y[:n_train], c='green', label='Training data')
plt.scatter(X[n_train:], y[n_train:], c='red', label='Test data')
plt.legend()
plt.show()

# 2) Model
# Start with random weights and biases. Look at training data and adjust weights and biases to get closer to the ideal values.
# How does it do this? It uses 2 main algorithms: gradient descent and backpropagation.
# Gradient descent is used to find the minimum of a function. In this case, the function is the loss function.
# Backpropagation is used to calculate the gradients of the loss function with respect to the weights and biases.
class LinearRegressionModel(nn.Module): # nn.Module is the base class for all neural network modules. LinearRegressionModel is a subclass of nn.Module. It inherits all the methods of nn.Module.
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) # super() is used to call the constructor of the parent class. In this case, the parent class is nn.Module.
        self.weight = nn.Parameter(torch.randn(1, torch.float)) # nn.Parameter is a wrapper for a tensor that tells a nn.Module that it has weights that need to be updated during training. 
        self.bias = nn.Parameter(torch.randn(1, torch.float)) # torch.randn returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1. The shape of the tensor is defined by the variable arguments. In this case, the shape is 1.

    def forward(self, x: torch.Tensor): # forward pass
        return self.weight * x + self.bias
    
    def predict(self, x: torch.Tensor): # predict method
        return self.forward(x)
    



