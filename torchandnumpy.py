import torch
import numpy as np

array = np.arange(1, 10)
print(array)
tensor = torch.from_numpy(array)
print(tensor)


# To reduce the randomness of the random number generator, we can use the torch.manual_seed() function.
# The torch.manual_seed() function sets the seed for generating random numbers.
# The torch.manual_seed() function takes a single argument, which is an integer.

# set the seed for generating random numbers
torch.manual_seed(42)
random_tensor = torch.rand((3, 5)) # 3 rows, 5 columns
print(random_tensor)

# why use random tensors?
# random tensors are useful for initializing weights in neural networks.
# why use seed?
# to reduce the randomness of the random number generator.
# how does a seed reduce the randomness of the random number generator?
# the seed sets the starting point for generating random numbers.


# How to run tensor using gpu?
# 1. Check if gpu is available
# 2. Move tensor to gpu

# check if gpu is available.
if torch.cuda.is_available():
    # move tensor to gpu
    tensor = tensor.to('cuda')
    print(tensor)
else:
    tensor = tensor.to('cpu')
    print("gpu is not available")

