#!/bin/env python

import torch
import numpy as np

data = [[1, 2], [3, 4]]
print(data)

# init from list
torch_data = torch.tensor(data)
print(torch_data)

# init from numpy
np_array = np.array(data)
print("numpy array:\n", np_array)
torch_data_from_numpy = torch.from_numpy(np_array)

# init from another tensor
torch_data2 = torch.ones_like(torch_data)
print(torch_data2)

torch_data3 = torch.rand_like(torch_data, dtype=torch.float)
print(torch_data3)

# init with value
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zero_tensor = torch.zeros(shape)
print(rand_tensor)
print(ones_tensor)
print(zero_tensor)
print(zero_tensor.dtype)

# devices
print(zero_tensor.device)

if torch.cuda.is_available():
    print("cuda available")
    print(zero_tensor.to('cuda').device)

# to numpy (share data memory)
zn = zero_tensor.numpy()
print(zn)
zero_tensor.add_(2)
print(zn)

n = np.zeros(5)
t = torch.from_numpy(n)
print(t)
np.add(n, 1, out=n)
print(t)

