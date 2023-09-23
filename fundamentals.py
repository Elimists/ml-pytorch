import torch

print("PyTorch version: " + torch.__version__)
"""
# what is a scalar?
# a scalar is a single number. It has magnitude but no direction.
scalar = torch.tensor(3.1415)
print(scalar)
print(scalar.item()) # 3.1415 <- get the value of the scalar

# what is a vector?
# a vector is a 1-dimensional tensor. It is a list of numbers. It has a direction and a magnitude.
vector = torch.tensor([1, 2])


# what is a matrix?
# a matrix is a 2-dimensional tensor. It is a list of lists.
matrix = torch.tensor([
                    [1, 2, 3], 
                    [4, 5, 6]
                ])
print(matrix)
print(matrix[1][1]) # 5
print(matrix.shape) # (2, 3) <- 2 rows, 3 columns
print(matrix.ndim) # 2 <- 2 dimensions

# what is meant by "shape"?
# shape is the dimensions of the tensor.

# -------------------------------------------------------------
# what is tensor?
# tensor is a multi-dimensional array

# what is torch.tensor?
# torch.tensor is a multi-dimensional array of single data type
tensor = torch.tensor([
    [[1, 2, 3], 
    [4, 5, 6], 
    [7, 8, 9]], # matrix 1 end
    [[10, 11, 12],
    [13, 14, 15],
    [16, 17, 18]] # matrix 2 end
    ]
    )
print(tensor.shape) # (2, 3, 3) <- 2 matrices, 3 rows, 3 columns
print("Dimensions: " + str(tensor.ndim)) # 3 <- 3 dimensions

#-------------------------------------------------------------
# what is a random tensor?
# a random tensor is a tensor with random values
random_tensor = torch.rand((3, 5)) # 3 rows, 5 columns
print(random_tensor)

# why use random tensors?
# random tensors are useful for initializing weights in neural networks. 
# Weights are the values that are multiplied by the input data. 
# The neural network learns the best weights to make accurate predictions. The weights are updated through training.

image_tensor = torch.rand((3, 224, 224)) # 3 channels, 224 rows (height), 224 columns(width). A 224x224 image with 3 channels (R,G,B).
print(image_tensor)
print(image_tensor.shape) # (3, 224, 224)
print(image_tensor.ndim) # 3
# -------------------------------------------------------------

# create tensor of all zeros (or ones)
zeros = torch.zeros((3, 3)) # 3 rows, 3 columns
print(zeros)
# why use tensors of zeros?
# tensors of zeros are useful for initializing bias terms in neural networks.
# Bias terms are values that are added to the output of each layer.
# The neural network learns the best bias terms to make accurate predictions and are updated through training.
# -------------------------------------------------------------

"""

aTensor = torch.tensor([
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                        ])
print(aTensor[1])

threeDimensionalTensor = torch.tensor([
                                        [[1, 2, 3], [4, 5, 6]],
                                        [[7, 8, 9], [10, 11, 12]]
                                        ])
print(threeDimensionalTensor[0][1])