import torch
# Create a tensor
tensor = torch.tensor([1,2,3])
print(tensor + 19)
print(tensor * 2)
print(tensor < 2)
print(tensor - 6)

print(tensor.min()) 
# OR
print(torch.min(tensor))

tensorB = torch.tensor([4,5,6])
print(tensor * tensorB) # element-wise multiplication.
print(torch.matmul(tensor, tensorB)) # matrix multiplication. 1*4 + 2*5 + 3*6 = 32
# when to use element-wise multiplication?
# element-wise multiplication is used when you want to multiply two tensors of the same shape (dimension).

# when to use matrix multiplication?
# matrix multiplication is used when you want to multiply two tensors of different shapes (dimensions).
# matrix multiplication is also used when you want to multiply a tensor with a matrix.

tensorC = torch.tensor([[1,2,3], [4,5,6]])
tensorD = torch.tensor([[7,8,12], [9,10,15]])
print("Element wise multiplication: " + str(tensorC * tensorD)) # element-wise multiplication  
try:
    print(torch.matmul(tensorC, tensorD))
except Exception as e:
    print("Error: Cannot perform matrix multiplication on tensors with shapes " + str(tensorC.shape) + " and " + str(tensorD.shape) + ". Must be of shape (m, n) and (n, p).")


#-------------------------------------------------------------

# what is the view operation?
# the view operation is an operation that changes the shape of a tensor without changing the underlying data.


# what is the reshape operation?
# the reshape operation is an operation that changes the shape of a tensor and changes the underlying data.


# what is the difference between view and reshape?
# view is used when you want to change the shape of a tensor without changing the underlying data.
# reshape is used when you want to change the shape of a tensor and change the underlying data.


# what is stacking?
# stacking is the process of joining a sequence of tensors along a new dimension.

# what is squeezing?
# squeezing is the process of removing a dimension from a tensor.

# what is unsqueezing?
# unsqueezing is the process of adding a dimension to a tensor.

# what is permuting?
# permuting is the process of changing the order of dimensions in a tensor.
original = torch.rand(220, 220, 3) # 220 rows, 220 columns, 3 channels. 3 channels is for RGB.
print(original.shape) # dimensions are (220, 220, 3)
print(original.ndim) # 3 dimensions
permuted = original.permute(2, 0, 1) # 3 channels, 220 rows, 220 columns. 
# how many ways can you permute a tensor?
# 6 ways. 3! = 3*2*1 = 6
# how many ways can you permute a tensor with 5 dimensions?
# 120 ways. 5! = 5*4*3*2*1 = 120
# how many ways can you permute a tensor with n dimensions?
# n! ways. n! = n*(n-1)*(n-2)*...*1

# what is flattening?
# flattening is the process of converting a tensor to a 1-dimensional tensor.
# what operation is used to flatten a tensor?
# the flatten operation is used to flatten a tensor.
# what is the difference between flatten and reshape?
# flatten converts a tensor to a 1-dimensional tensor.
# reshape converts a tensor to a tensor of any
flattened = original.flatten()
print(flattened.shape) # (145200,)
flattened_permuted = permuted.flatten()
print(flattened_permuted.shape) # (145200,)
"""
# create a tensor
x = torch.arange(1, 10)
print(x, x.shape)

#Change the view
changed_view_x = x.view(3, 3) # change the shape of x to 3 rows, 3 columns
print(changed_view_x, changed_view_x.shape)
# change data in changed_view_x
changed_view_x[0][0] = 99
print(changed_view_x, changed_view_x.shape)
# print x
print(x) 

# Reshape
reshaped_x = x.reshape(3, 3) # reshape x to 3 rows, 3 columns
print(reshaped_x, reshaped_x.shape)
# change data in reshaped_x
reshaped_x[0][0] = 99
print(reshaped_x, reshaped_x.shape)
# print x
print(x) # x is changed to 99, 2, 3, 4, 5, 6, 7, 8, 9

a = torch.tensor([[[1,2,3]],[[4,5,6]]]) 
print(a, a.shape) # a is a 2-dimensional tensor
y = torch.squeeze(a) # remove the dimension of size 1 from x
print(y, y.shape) # y is now a 1-dimensional tensor

"""
